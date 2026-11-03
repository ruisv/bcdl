#include "bcdl/tasks/tts.h"

#include <algorithm>
#include <utility>
#include <cmath>
#include <fstream>
#include <sstream>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/input_writer.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"
#include "task_util.h"
#include "bcdl/tasks/vocoder.h"
#include "cn_number.h"

namespace bcdl {

namespace {

// --- UTF-8 ------------------------------------------------------------------

std::u32string toU32(const std::string& s) {
  std::u32string out;
  for (std::size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
    if (len == 0 || i + len > s.size()) {
      ++i;  // malformed byte: skip
      continue;
    }
    char32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
    bool ok = true;
    for (int k = 1; k < len; ++k) {
      const unsigned char b = static_cast<unsigned char>(s[i + k]);
      if ((b & 0xC0) != 0x80) ok = false;
      cp = (cp << 6) | (b & 0x3F);
    }
    i += ok ? len : 1;
    if (ok) out.push_back(cp);
  }
  return out;
}

std::string toU8(const std::u32string& s) {
  std::string out;
  for (char32_t cp : s) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }
  return out;
}

// --- character classes (as sherpa-onnx's text-utils) -------------------------

bool isCjk(char32_t c) {
  return (c >= 0x1100 && c <= 0x11FF) || (c >= 0x2E80 && c <= 0xA4CF) ||
         (c >= 0xA840 && c <= 0xD7AF) || (c >= 0xF900 && c <= 0xFAFF) ||
         (c >= 0xFE30 && c <= 0xFE4F) || (c >= 0xFF65 && c <= 0xFFDC) ||
         (c >= 0x20000 && c <= 0x2FFFF);
}
bool isSpace(char32_t c) {
  return c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' || c == U'\f' || c == U'\v';
}
bool isSentenceEnd(char32_t c) {
  return c == U'.' || c == U'!' || c == U'?' || c == U'。' || c == U'！' || c == U'？';
}
bool isChunkEnd(char32_t c) {
  return isSentenceEnd(c) || c == U',' || c == U';' || c == U':' || c == U'，' || c == U'；' ||
         c == U'：';
}
bool isPunct(char32_t c) {
  return c == U',' || c == U'.' || c == U';' || c == U':' || c == U'!' || c == U'?';
}
bool isDigit(char32_t c) { return c >= U'0' && c <= U'9'; }

std::u32string trim(const std::u32string& s) {
  std::size_t a = 0, b = s.size();
  while (a < b && isSpace(s[a])) ++a;
  while (b > a && isSpace(s[b - 1])) --b;
  return s.substr(a, b - a);
}

int countWords(const std::u32string& s) {
  int n = 0;
  bool in_word = false;
  for (char32_t c : s) {
    if (isCjk(c) || isSentenceEnd(c)) {
      ++n;
      in_word = false;
    } else if (isSpace(c) || c == U',' || c == U';' || c == U':' || c == U'，' || c == U'；' ||
               c == U'：') {
      in_word = false;
    } else if (!in_word) {
      ++n;
      in_word = true;
    }
  }
  return n;
}

/// Chinese punctuation -> ASCII, as sherpa-onnx's ReplacePunctuations.
std::u32string foldPunctuation(const std::u32string& s) {
  std::u32string out;
  for (char32_t c : s) {
    switch (c) {
      case U'，': case U'、': case U'：': case U':': out += U','; break;
      case U'；': out += U';'; break;
      case U'。': out += U'.'; break;
      case U'？': out += U'?'; break;
      case U'！': out += U'!'; break;
      case U'…': out += U"..."; break;
      default: out += c;
    }
  }
  return out;
}

/// Split after every chunk boundary (the punctuation stays with its chunk).
std::vector<std::u32string> splitChunks(const std::u32string& text) {
  std::vector<std::u32string> out;
  std::u32string cur;
  auto flush = [&] {
    std::u32string t = trim(cur);
    if (!t.empty()) out.push_back(t);
    cur.clear();
  };
  for (char32_t c : text) {
    cur.push_back(c);
    if (isChunkEnd(c)) flush();
  }
  flush();
  return out;
}

std::vector<std::u32string> mergeShort(const std::vector<std::u32string>& in, int min_words) {
  std::vector<std::u32string> out;
  std::u32string buf;
  for (const auto& s : in) {
    std::u32string p = trim(s);
    if (p.empty()) continue;
    buf += p;
    if (countWords(buf) >= min_words) {
      out.push_back(trim(buf));
      buf.clear();
    }
  }
  if (!buf.empty()) out.push_back(trim(buf));
  return out;
}

std::vector<std::u32string> splitLong(const std::u32string& sentence, int max_words) {
  std::vector<std::u32string> out;
  const std::u32string s = trim(sentence);
  const std::size_t len = s.size();
  std::size_t start = 0;
  while (start < len) {
    int words = 0;
    bool in_word = false;
    std::size_t end = start;
    for (std::size_t i = start; i < len; ++i) {
      const char32_t c = s[i];
      if (isCjk(c) || isSentenceEnd(c)) {
        ++words;
        in_word = false;
      } else if (isSpace(c) || c == U',' || c == U';' || c == U':' || c == U'，' ||
                 c == U'；' || c == U'：') {
        in_word = false;
      } else if (!in_word) {
        ++words;
        in_word = true;
      }
      if (words > max_words) break;
      end = i + 1;
    }
    if (end >= len) {
      std::u32string p = trim(s.substr(start));
      if (!p.empty()) out.push_back(p);
      break;
    }
    std::size_t split = end;
    bool found = false;
    for (std::size_t i = end; i > start; --i) {
      const char32_t c = s[i - 1];
      if (isSpace(c)) {
        split = i - 1;
        found = true;
        break;
      }
      if (isChunkEnd(c)) {
        split = i;
        found = true;
        break;
      }
    }
    if (!found || split <= start) split = end;
    std::u32string p = trim(s.substr(start, split - start));
    if (!p.empty()) out.push_back(p);
    start = split;
    while (start < len && isSpace(s[start])) ++start;
  }
  return out;
}

std::string lower(std::string s) {
  for (auto& c : s)
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  return s;
}

/// Lookup units of a lexicon key: one per CJK character, space, punctuation
/// mark or other run (sherpa's CountLexiconPhraseWords).
int phraseWords(const std::u32string& k) {
  int n = 0;
  bool in_run = false;
  for (char32_t c : k) {
    if (isCjk(c) || isSpace(c) || isPunct(c)) {
      ++n;
      in_run = false;
    } else if (!in_run) {
      ++n;
      in_run = true;
    }
  }
  return n;
}

}  // namespace

// --- front end ----------------------------------------------------------------

MatchaFrontend::MatchaFrontend(const std::string& tokens, const std::string& lexicon,
                               MatchaFrontendConfig cfg)
    : cfg_(cfg) {
  if (cfg_.min_words < 1 || cfg_.max_words < 1)
    throw Error(-1, "BCDL MatchaFrontend: min_words and max_words must be >= 1");
  std::ifstream tf(tokens);
  if (!tf) throw Error(-1, "BCDL MatchaFrontend: cannot open tokens " + tokens);
  std::string line;
  while (std::getline(tf, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::istringstream is(line);
    std::vector<std::string> f;
    for (std::string w; is >> w;) f.push_back(w);
    if (f.empty() || f.size() > 2) continue;
    // The space token is written as a bare id.
    const std::string& id_str = f.back();
    if (id_str.find_first_not_of("0123456789") != std::string::npos || id_str.size() > 9)
      throw Error(-1, "BCDL MatchaFrontend: bad token id '" + id_str + "' in " + tokens);
    const int id = std::stoi(id_str);
    token_[f.size() == 1 ? std::string(" ") : f[0]] = id;
    vocab_ = std::max(vocab_, id + 1);
  }
  if (token_.empty()) throw Error(-1, "BCDL MatchaFrontend: no tokens in " + tokens);
  for (const char* p : {" ", ",", ".", ";", ":", "!", "?"}) {
    auto it = token_.find(p);
    if (it != token_.end()) silent_.insert(it->second);
  }

  std::ifstream lf(lexicon);
  if (!lf) throw Error(-1, "BCDL MatchaFrontend: cannot open lexicon " + lexicon);
  while (std::getline(lf, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#' || line.find("||") != std::string::npos) continue;
    std::istringstream is(line);
    std::string word;
    if (!(is >> word)) continue;
    std::vector<int> ids;
    for (std::string ph; is >> ph;) {
      auto it = token_.find(ph);
      if (it != token_.end()) ids.push_back(it->second);  // unknown phones are skipped
    }
    if (ids.empty()) continue;
    const std::string key = lower(word);
    max_phrase_ = std::max(max_phrase_, phraseWords(toU32(key)));
    lexicon_[key] = std::move(ids);
  }
  if (lexicon_.empty()) throw Error(-1, "BCDL MatchaFrontend: no usable entries in " + lexicon);
}

std::string MatchaFrontend::readNumbers(const std::string& text) {
  const std::u32string s = toU32(text);
  std::u32string out;
  std::size_t i = 0;
  while (i < s.size()) {
    if (!isDigit(s[i])) {
      out.push_back(s[i++]);
      continue;
    }
    std::size_t j = i;
    while (j < s.size() && isDigit(s[j])) ++j;
    std::string ip(s.begin() + i, s.begin() + j);  // ASCII digits
    std::string frac;
    std::size_t k = j;
    if (k + 1 < s.size() && s[k] == U'.' && isDigit(s[k + 1])) {
      ++k;
      while (k < s.size() && isDigit(s[k])) frac.push_back(static_cast<char>(s[k++]));
    }
    const bool percent = k < s.size() && (s[k] == U'%' || s[k] == U'％');
    std::string spoken;
    if (!percent && frac.empty() && ip.size() == 4 && k < s.size() && s[k] == U'年') {
      spoken = detail::cnDigits(ip);  // 2026年 -> 二零二六年
    } else {
      spoken = detail::cnNumber(ip);
      if (!frac.empty()) spoken += "点" + detail::cnDigits(frac);
    }
    if (percent) {
      spoken = "百分之" + spoken;
      ++k;
    }
    out += toU32(spoken);
    i = k;
  }
  return toU8(out);
}

std::vector<std::string> MatchaFrontend::sentences(const std::string& text) const {
  const std::u32string norm = foldPunctuation(toU32(readNumbers(text)));
  std::vector<std::u32string> pieces;
  for (const auto& s : mergeShort(splitChunks(norm), cfg_.min_words)) {
    auto p = splitLong(s, cfg_.max_words);
    pieces.insert(pieces.end(), p.begin(), p.end());
  }
  // A punctuation-only piece joins the previous sentence.
  std::vector<std::string> out;
  for (const auto& p : pieces) {
    const std::u32string t = trim(p);
    bool all_punct = !t.empty();
    for (char32_t c : t)
      if (!isPunct(c) && c != U' ') all_punct = false;
    if (all_punct && !out.empty()) {
      out.back() += toU8(t);
    } else {
      out.push_back(toU8(p));
    }
  }
  return out;
}

std::vector<int> MatchaFrontend::tokenize(const std::string& sentence) const {
  // Words: a space, a punctuation mark and a CJK character each stand alone;
  // anything else accumulates into a run.
  const std::u32string u = toU32(sentence);
  std::vector<std::u32string> words;
  std::u32string run;
  auto flush = [&] {
    if (!run.empty()) words.push_back(run);
    run.clear();
  };
  for (char32_t c : u) {
    if (c == U' ' || c == U'\t' || c == U'\n' || c == U'\r') {
      flush();
      words.push_back(U" ");
    } else if (isPunct(c) || isCjk(c)) {
      flush();
      words.push_back(std::u32string(1, c));
    } else {
      run.push_back(c);
    }
  }
  flush();

  std::vector<int> ids;
  const int n = static_cast<int>(words.size());
  for (int i = 0; i < n;) {
    if (words[i] == U" ") {
      auto it = token_.find(" ");
      if (it != token_.end()) ids.push_back(it->second);
      ++i;
      continue;
    }
    if (words[i].size() == 1 && isPunct(words[i][0])) {
      auto it = token_.find(toU8(words[i]));
      if (it != token_.end()) ids.push_back(it->second);
      ++i;
      continue;
    }
    // Longest match: CJK neighbours join directly, anything else with a space.
    bool found = false;
    for (int len = std::min(max_phrase_, n - i); len >= 1; --len) {
      std::u32string phrase;
      for (int j = i; j < i + len; ++j) {
        const bool prev_cjk = !phrase.empty() && isCjk(phrase.back());
        const bool cur_cjk = !words[j].empty() && isCjk(words[j].front());
        if (j > i && !(prev_cjk && cur_cjk)) phrase.push_back(U' ');
        phrase += words[j];
      }
      auto it = lexicon_.find(lower(toU8(phrase)));
      if (it != lexicon_.end()) {
        ids.insert(ids.end(), it->second.begin(), it->second.end());
        i += len;
        found = true;
        break;
      }
    }
    if (!found) {
      // CJK not in the lexicon as a whole: character by character. Anything
      // else (Latin words: no English in this lexicon) is skipped.
      for (char32_t c : words[i]) {
        if (!isCjk(c)) continue;
        auto it = lexicon_.find(toU8(std::u32string(1, c)));
        if (it != lexicon_.end()) ids.insert(ids.end(), it->second.begin(), it->second.end());
      }
      ++i;
    }
  }
  return ids;
}

std::vector<std::vector<int>> MatchaFrontend::encode(const std::string& text) const {
  std::vector<std::vector<int>> out;
  for (const auto& s : sentences(text)) {
    std::vector<int> ids = tokenize(s);
    // The sentence's closing punctuation is a token too.
    const std::u32string t = trim(toU32(s));
    if (!t.empty() && !ids.empty() && isPunct(t.back())) {
      auto it = token_.find(toU8(std::u32string(1, t.back())));
      if (it != token_.end() && ids.back() != it->second) ids.push_back(it->second);
    }
    // A sentence with nothing to say (punctuation and spaces only) is dropped.
    bool speech = false;
    for (int id : ids)
      if (!silent_.count(id)) speech = true;
    if (speech) out.push_back(std::move(ids));
  }
  return out;
}

// --- acoustic model -------------------------------------------------------------

MatchaAcoustic::MatchaAcoustic(Engine& encoder, Engine& decoder, const std::string& embedding,
                               MatchaConfig cfg)
    : enc_(encoder), dec_(decoder), cfg_(cfg), rng_(cfg.seed) {
  int rows = 0, cols = 0;
  table_ = detail::loadNpy2d(embedding, rows, cols, "MatchaAcoustic");
  vocab_ = rows;
  dim_ = cols;
  init();
}

MatchaAcoustic::MatchaAcoustic(Engine& encoder, Engine& decoder, std::vector<float> table,
                               int vocab, MatchaConfig cfg)
    : enc_(encoder), dec_(decoder), cfg_(cfg), rng_(cfg.seed), table_(std::move(table)),
      vocab_(vocab) {
  if (vocab_ <= 0 || table_.empty() || table_.size() % vocab_ != 0)
    throw Error(-1, "BCDL MatchaAcoustic: embedding table size is not a multiple of vocab");
  dim_ = static_cast<int>(table_.size() / vocab_);
  init();
}

void MatchaAcoustic::init() {
  if (enc_.numInputs() != 2 || enc_.numOutputs() != 2)
    throw Error(-1, "BCDL MatchaAcoustic: encoder must have 2 inputs (x_emb, x_mask) and 2 "
                    "outputs (mu_x, logw)");
  if (dec_.numInputs() != 3 || dec_.numOutputs() != 1)
    throw Error(-1, "BCDL MatchaAcoustic: decoder must have 3 inputs (mu_y, y_mask, z) and 1 "
                    "output (mel)");
  // Encoder: x_emb [1, D, L] vs x_mask [1, 1, L]; mu_x [1, mels, L] vs logw [1, 1, L].
  for (int i = 0; i < 2; ++i) {
    const auto s = enc_.inputShape(i);
    if (s.size() == 3 && s[1] == 1) enc_mask_ = i;
    else enc_tok_ = i;
  }
  L_ = enc_.inputShape(enc_tok_).back();
  if (enc_.inputShape(enc_tok_) != std::vector<int>{1, dim_, L_})
    throw Error(-1, "BCDL MatchaAcoustic: encoder input is " +
                        detail::shapeStr(enc_.inputShape(enc_tok_)) + ", the embedding table is " +
                        std::to_string(vocab_) + " x " + std::to_string(dim_));
  for (int i = 0; i < 2; ++i) {
    const auto s = enc_.outputShape(i);
    if (s.size() == 3 && s[1] == 1) enc_logw_ = i;
    else enc_mu_ = i;
  }
  mels_ = enc_.outputShape(enc_mu_)[1];
  if (enc_.inputShape(enc_mask_) != std::vector<int>{1, 1, L_} ||
      enc_.outputShape(enc_mu_) != std::vector<int>{1, mels_, L_} ||
      enc_.outputShape(enc_logw_) != std::vector<int>{1, 1, L_})
    throw Error(-1, "BCDL MatchaAcoustic: encoder shapes disagree: tokens " +
                        detail::shapeStr(enc_.inputShape(enc_tok_)) + ", mu_x " +
                        detail::shapeStr(enc_.outputShape(enc_mu_)));
  // Decoder: y_mask is the [1, 1, T] input. mu_y and z have the same shape,
  // so they are told apart by name when the build carries the export's names
  // (mu_y, z), and by order otherwise.
  std::vector<int> wide;
  for (int i = 0; i < 3; ++i) {
    const auto s = dec_.inputShape(i);
    if (s.size() == 3 && s[1] == 1) dec_mask_ = i;
    else wide.push_back(i);
  }
  if (wide.size() != 2)
    throw Error(-1, "BCDL MatchaAcoustic: decoder needs one [1,1,T] mask input");
  dec_mu_ = wide[0];
  dec_z_ = wide[1];
  {
    const std::string a = dec_.inputName(wide[0]), b = dec_.inputName(wide[1]);
    if (a == "z" && b == "mu_y") std::swap(dec_mu_, dec_z_);
    else if (!(a == "mu_y" && b == "z") && (a == "z" || b == "z" || a == "mu_y" || b == "mu_y"))
      throw Error(-1, "BCDL MatchaAcoustic: decoder inputs are named '" + a + "' and '" + b +
                          "', expected mu_y and z");
  }
  T_ = dec_.inputShape(dec_mask_).back();
  for (int i : {dec_mu_, dec_z_})
    if (dec_.inputShape(i) != std::vector<int>{1, mels_, T_})
      throw Error(-1, "BCDL MatchaAcoustic: decoder input " + std::to_string(i) + " is " +
                          detail::shapeStr(dec_.inputShape(i)) + ", expected [1, " +
                          std::to_string(mels_) + ", " + std::to_string(T_) + "]");
  if (dec_.outputShape(0) != std::vector<int>{1, mels_, T_})
    throw Error(-1, "BCDL MatchaAcoustic: decoder output is " + detail::shapeStr(dec_.outputShape(0)));
  if (!(cfg_.length_scale > 0.0f) || !(cfg_.noise_scale >= 0.0f))
    throw Error(-1, "BCDL MatchaAcoustic: length_scale must be > 0 and noise_scale >= 0");
}

MatchaResult MatchaAcoustic::synthesize(const int* ids, int n, int timeout_ms) {
  MatchaResult r;
  if (n <= 0) return r;
  if (ids == nullptr) throw Error(-1, "BCDL MatchaAcoustic::synthesize: null ids");
  if (n > L_) {
    r.truncated = true;
    n = L_;
  }
  emb_.assign(static_cast<std::size_t>(dim_) * L_, 0.0f);
  xmask_.assign(L_, 0.0f);
  for (int i = 0; i < n; ++i) {
    if (ids[i] < 0 || ids[i] >= vocab_)
      throw Error(-1, "BCDL MatchaAcoustic: token id " + std::to_string(ids[i]) +
                          " outside the " + std::to_string(vocab_) + "-entry table");
    const float* row = table_.data() + static_cast<std::size_t>(ids[i]) * dim_;
    for (int d = 0; d < dim_; ++d) emb_[static_cast<std::size_t>(d) * L_ + i] = row[d];
    xmask_[i] = 1.0f;
  }
  setInputFromFloat(enc_, enc_tok_, emb_.data(), emb_.size(), stage_);
  setInputFromFloat(enc_, enc_mask_, xmask_.data(), xmask_.size(), stage_);
  enc_.infer(timeout_ms);
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* mu = outputAsFloat(enc_, enc_mu_, scratch, shape);
  mu_x_.assign(mu, mu + static_cast<std::size_t>(mels_) * L_);
  const float* lw = outputAsFloat(enc_, enc_logw_, scratch, shape);
  logw_.assign(lw, lw + L_);

  // Durations exactly as Matcha: ceil(exp(logw)) * length_scale; token i covers
  // frames [cum_{i-1}, cum_i), and the length is the floored sum (at least 1).
  std::vector<double> cum(n);
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    total += std::ceil(std::exp(static_cast<double>(logw_[i]))) * cfg_.length_scale;
    cum[i] = total;
  }
  int frames = std::max(1, static_cast<int>(total));
  r.durations.resize(n);
  {
    int prev = 0;
    for (int i = 0; i < n; ++i) {
      const int upto = std::min(frames, static_cast<int>(std::ceil(cum[i])));
      r.durations[i] = std::max(0, upto - prev);
      prev = std::max(prev, upto);
    }
  }
  if (frames > T_) {
    r.truncated = true;
    frames = T_;
  }

  mu_y_.assign(static_cast<std::size_t>(mels_) * T_, 0.0f);
  ymask_.assign(T_, 0.0f);
  z_.assign(mu_y_.size(), 0.0f);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  int tokn = 0;
  for (int t = 0; t < frames; ++t) {
    while (tokn < n - 1 && static_cast<double>(t) >= cum[tokn]) ++tokn;
    for (int c = 0; c < mels_; ++c)
      mu_y_[static_cast<std::size_t>(c) * T_ + t] = mu_x_[static_cast<std::size_t>(c) * L_ + tokn];
    ymask_[t] = 1.0f;
  }
  for (int c = 0; c < mels_; ++c)
    for (int t = 0; t < frames; ++t)
      z_[static_cast<std::size_t>(c) * T_ + t] = normal(rng_) * cfg_.noise_scale;

  setInputFromFloat(dec_, dec_mu_, mu_y_.data(), mu_y_.size(), stage_);
  setInputFromFloat(dec_, dec_mask_, ymask_.data(), ymask_.size(), stage_);
  setInputFromFloat(dec_, dec_z_, z_.data(), z_.size(), stage_);
  dec_.infer(timeout_ms);
  const float* mel = outputAsFloat(dec_, 0, scratch, shape);
  r.frames = frames;
  r.mel.resize(static_cast<std::size_t>(mels_) * frames);
  for (int c = 0; c < mels_; ++c)
    std::copy_n(mel + static_cast<std::size_t>(c) * T_, frames,
                r.mel.data() + static_cast<std::size_t>(c) * frames);
  return r;
}

// --- text to speech ----------------------------------------------------------------

std::vector<float> scaleSilence(const std::vector<float>& pcm, int sample_rate, float scale,
                                float min_pause_s, float level) {
  if (!(scale >= 0.01f && scale <= 10.0f) || scale == 1.0f) return pcm;
  const std::size_t n = pcm.size();
  const std::size_t min_len = static_cast<std::size_t>(sample_rate * min_pause_s);
  std::vector<float> out;
  out.reserve(scale < 1.0f ? n : static_cast<std::size_t>(n * scale));
  std::size_t i = 0, copied = 0;
  while (i < n) {
    if (std::fabs(pcm[i]) > level) {
      ++i;
      continue;
    }
    std::size_t j = i;
    while (j < n && std::fabs(pcm[j]) <= level) ++j;
    // As sherpa-onnx: an inner pause counts from min_len samples, a trailing one past it.
    if (j < n ? j - i >= min_len : j - i > min_len) {
      out.insert(out.end(), pcm.begin() + copied, pcm.begin() + i);
      const std::size_t keep = static_cast<std::size_t>((j - i) * static_cast<double>(scale));
      const std::size_t from_input = std::min(keep, j - i);
      out.insert(out.end(), pcm.begin() + i, pcm.begin() + i + from_input);
      out.insert(out.end(), keep - from_input, 0.0f);
      copied = j;
    }
    i = j;
  }
  out.insert(out.end(), pcm.begin() + copied, pcm.end());
  return out;
}

TextToSpeech::TextToSpeech(MatchaFrontend frontend, MatchaAcoustic& acoustic,
                           VocosVocoder& vocoder, TextToSpeechConfig cfg)
    : fe_(std::move(frontend)), cfg_(cfg), am_(acoustic), voc_(vocoder) {
  if (am_.numMels() != voc_.numMels())
    throw Error(-1, "BCDL TextToSpeech: acoustic model emits " + std::to_string(am_.numMels()) +
                        " mels, the vocoder takes " + std::to_string(voc_.numMels()));
}

TtsResult TextToSpeech::synthesize(const std::string& text, int timeout_ms) {
  TtsResult r;
  r.sample_rate = voc_.sampleRate();
  std::vector<float> pcm;
  for (const auto& ids : fe_.encode(text)) {
    MatchaResult m = am_.synthesize(ids.data(), static_cast<int>(ids.size()), timeout_ms);
    r.truncated |= m.truncated;
    if (m.frames == 0) continue;
    voc_.synthesize(m.mel.data(), m.frames, pcm, timeout_ms);
    r.pcm.insert(r.pcm.end(), pcm.begin(), pcm.end());
    ++r.sentences;
  }
  r.pcm = scaleSilence(r.pcm, r.sample_rate, cfg_.silence_scale);
  return r;
}

}  // namespace bcdl
