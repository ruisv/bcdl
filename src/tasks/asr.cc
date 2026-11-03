#include "bcdl/tasks/asr.h"

#include <algorithm>
#include <fstream>
#include <sstream>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/input_writer.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"
#include "task_util.h"

namespace bcdl {

namespace {

/// Append one UTF-8 encoded code point.
void putUtf8(std::string& s, unsigned cp) {
  if (cp < 0x80) {
    s += static_cast<char>(cp);
  } else if (cp < 0x800) {
    s += static_cast<char>(0xC0 | (cp >> 6));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    s += static_cast<char>(0xE0 | (cp >> 12));
    s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    s += static_cast<char>(0xF0 | (cp >> 18));
    s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

/// Parse a JSON array of strings. Deliberately just that: a token list is the
/// only JSON this library reads, and a general parser is not worth carrying.
std::vector<std::string> parseJsonStringArray(const std::string& t, const std::string& path) {
  std::vector<std::string> out;
  std::size_t i = t.find('[');
  auto fail = [&](const char* why) {
    throw Error(-1, std::string("BCDL loadTokenList: ") + why + " in " + path);
  };
  if (i == std::string::npos) fail("no '['");
  ++i;
  auto skipWs = [&] {
    while (i < t.size() && (t[i] == ' ' || t[i] == '\n' || t[i] == '\r' || t[i] == '\t')) ++i;
  };
  skipWs();
  if (i < t.size() && t[i] == ']') return out;
  while (true) {
    skipWs();
    if (i >= t.size() || t[i] != '"') fail("expected a string");
    ++i;
    std::string s;
    while (true) {
      if (i >= t.size()) fail("unterminated string");
      const char c = t[i++];
      if (c == '"') break;
      if (c != '\\') {
        s += c;
        continue;
      }
      if (i >= t.size()) fail("dangling escape");
      const char e = t[i++];
      switch (e) {
        case '"': s += '"'; break;
        case '\\': s += '\\'; break;
        case '/': s += '/'; break;
        case 'b': s += '\b'; break;
        case 'f': s += '\f'; break;
        case 'n': s += '\n'; break;
        case 'r': s += '\r'; break;
        case 't': s += '\t'; break;
        case 'u': {
          auto hex4 = [&]() -> unsigned {
            if (i + 4 > t.size()) fail("short \\u escape");
            unsigned v = 0;
            for (int k = 0; k < 4; ++k) {
              const char h = t[i++];
              v <<= 4;
              if (h >= '0' && h <= '9') v |= h - '0';
              else if (h >= 'a' && h <= 'f') v |= h - 'a' + 10;
              else if (h >= 'A' && h <= 'F') v |= h - 'A' + 10;
              else fail("bad \\u escape");
            }
            return v;
          };
          unsigned cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00) {
            if (!(i + 1 < t.size() && t[i] == '\\' && t[i + 1] == 'u')) fail("lone surrogate");
            i += 2;
            const unsigned lo = hex4();
            if (lo < 0xDC00 || lo > 0xDFFF) fail("bad low surrogate");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            fail("lone surrogate");
          }
          putUtf8(s, cp);
          break;
        }
        default: fail("unknown escape");
      }
    }
    out.push_back(std::move(s));
    skipWs();
    if (i < t.size() && t[i] == ',') {
      ++i;
      continue;
    }
    if (i < t.size() && t[i] == ']') break;
    fail("expected ',' or ']'");
  }
  return out;
}

bool isSpecial(const std::string& tok) {
  return tok.size() >= 2 && tok.front() == '<' && tok.back() == '>';
}

bool isLatinWord(const std::string& s) {
  if (s.empty()) return false;
  for (unsigned char c : s) {
    if (c >= 0x80) return false;
  }
  return std::any_of(s.begin(), s.end(),
                     [](unsigned char c) { return (c | 0x20) >= 'a' && (c | 0x20) <= 'z'; });
}

bool shapeIs(const std::vector<int>& s, std::initializer_list<int> want) {
  return s.size() == want.size() && std::equal(s.begin(), s.end(), want.begin());
}

}  // namespace

int cifIntegrate(const float* alphas, const float* hidden, int frames, int dim,
                 int valid_frames, const CifConfig& cfg, std::vector<float>& embeds) {
  if (alphas == nullptr || hidden == nullptr || frames <= 0 || dim <= 0) {
    throw Error(-1, "BCDL cifIntegrate: null or empty predictor output");
  }
  if (cfg.max_tokens <= 0 || !(cfg.threshold > 0)) {
    throw Error(-1, "BCDL cifIntegrate: max_tokens and threshold must be > 0");
  }
  if (valid_frames < 0 || valid_frames > frames) {
    throw Error(-1, "BCDL cifIntegrate: valid_frames " + std::to_string(valid_frames) +
                        " outside [0, " + std::to_string(frames) + "]");
  }
  const bool tail = cfg.tail_threshold > 0;
  if (tail && valid_frames >= frames) {
    throw Error(-1, "BCDL cifIntegrate: no frame left after valid_frames for the tail weight");
  }
  embeds.assign(static_cast<std::size_t>(cfg.max_tokens) * dim, 0.0f);

  // The FunASR loop, frame by frame: accumulate weight * hidden until the
  // running sum crosses the threshold, emit, and carry the overshoot of that
  // frame into the next token. At most one fire per frame — every Paraformer
  // keeps each alpha below 1.
  const int last = valid_frames + (tail ? 1 : 0);
  std::vector<double> frame(dim, 0.0);
  double integrate = 0.0;
  int count = 0;
  for (int t = 0; t < last; ++t) {
    const bool is_tail = t == valid_frames;
    const double a = is_tail ? cfg.tail_threshold : std::max(0.0f, alphas[t]);
    const float* h = hidden + static_cast<std::size_t>(t) * dim;
    const double completion = 1.0 - integrate;
    integrate += a;
    const bool fire = integrate >= cfg.threshold;
    const double cur = fire ? completion : a;
    for (int d = 0; d < dim; ++d) frame[d] += cur * (is_tail ? 0.0 : h[d]);
    if (fire) {
      if (count < cfg.max_tokens) {
        float* o = embeds.data() + static_cast<std::size_t>(count) * dim;
        for (int d = 0; d < dim; ++d) o[d] = static_cast<float>(frame[d]);
      }
      ++count;
      integrate -= 1.0;
      const double rem = a - cur;
      for (int d = 0; d < dim; ++d) frame[d] = rem * (is_tail ? 0.0 : h[d]);
    }
  }
  return std::min(count, cfg.max_tokens);
}

std::vector<std::string> loadTokenList(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error(-1, "BCDL loadTokenList: cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string t = ss.str();
  const std::size_t first = t.find_first_not_of(" \t\r\n\xEF\xBB\xBF");
  std::vector<std::string> out;
  if (first != std::string::npos && t[first] == '[') {
    out = parseJsonStringArray(t, path);
  } else {
    // A UTF-8 BOM would otherwise stick to the first token (usually <blank>).
    const bool bom = t.compare(0, 3, "\xEF\xBB\xBF") == 0;
    std::istringstream in(bom ? t.substr(3) : t);
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      // "token<space>id" (k2/sherpa tokens.txt) or a bare token per line.
      const std::size_t sp = line.rfind(' ');
      if (sp != std::string::npos && sp + 1 < line.size() &&
          line.find_first_not_of("0123456789", sp + 1) == std::string::npos) {
        line.resize(sp);
      }
      if (!line.empty()) out.push_back(line);
    }
  }
  if (out.empty()) throw Error(-1, "BCDL loadTokenList: no tokens in " + path);
  return out;
}

std::vector<int> argmaxTokens(const float* logits, int rows, int vocab) {
  if (rows < 0 || vocab <= 0) throw Error(-1, "BCDL argmaxTokens: bad shape");
  std::vector<int> ids(rows);
  if (rows > 0 && logits == nullptr) throw Error(-1, "BCDL argmaxTokens: null logits");
  for (int r = 0; r < rows; ++r) {
    const float* row = logits + static_cast<std::size_t>(r) * vocab;
    ids[r] = static_cast<int>(std::max_element(row, row + vocab) - row);
  }
  return ids;
}

std::string tokensToText(const std::vector<int>& ids, const std::vector<std::string>& vocab) {
  std::string text;
  bool glue = false;        // previous piece ended in "@@"
  bool prev_latin = false;  // previous piece was a Latin word
  for (int id : ids) {
    if (id < 0 || id >= static_cast<int>(vocab.size())) {
      throw Error(-1, "BCDL tokensToText: token id " + std::to_string(id) +
                          " outside a vocabulary of " + std::to_string(vocab.size()));
    }
    std::string tok = vocab[id];
    if (isSpecial(tok)) continue;
    bool next_glue = false;
    if (tok.size() >= 2 && tok.compare(tok.size() - 2, 2, "@@") == 0) {
      tok.resize(tok.size() - 2);
      next_glue = true;
    }
    const bool latin = isLatinWord(tok);
    if (!glue && latin && prev_latin) text += ' ';
    text += tok;
    glue = next_glue;
    prev_latin = latin;
  }
  return text;
}

ParaformerAsr::ParaformerAsr(Engine& encoder, Engine& predictor, Engine& decoder,
                             std::vector<std::string> vocab, Cmvn cmvn, ParaformerConfig cfg)
    : enc_(encoder),
      pred_(predictor),
      dec_(decoder),
      vocab_(std::move(vocab)),
      cmvn_(std::move(cmvn)),
      cfg_(cfg),
      fbank_(cfg.fbank) {
  const char* who = "BCDL ParaformerAsr: ";
  // Encoder: [1,T,F] -> [1,T,D].
  if (enc_.numInputs() != 1 || enc_.numOutputs() < 1) {
    throw Error(-1, std::string(who) + "encoder must have 1 input and an output");
  }
  const std::vector<int> ei = enc_.inputShape(0), eo = enc_.outputShape(0);
  if (ei.size() != 3 || ei[0] != 1 || eo.size() != 3 || eo[0] != 1 || eo[1] != ei[1]) {
    throw Error(-1, std::string(who) + "encoder wants [1,T,F] -> [1,T,D], got " +
                        detail::shapeStr(ei) + " -> " + detail::shapeStr(eo));
  }
  frames_ = ei[1];
  feat_dim_ = ei[2];
  hid_dim_ = eo[2];
  if (feat_dim_ != cfg_.fbank.num_mel_bins * cfg_.lfr_m) {
    throw Error(-1, std::string(who) + "encoder feature dim " + std::to_string(feat_dim_) +
                        " != num_mel_bins * lfr_m = " +
                        std::to_string(cfg_.fbank.num_mel_bins * cfg_.lfr_m));
  }
  if (cmvn_.dim() != feat_dim_) {
    throw Error(-1, std::string(who) + "CMVN has " + std::to_string(cmvn_.dim()) +
                        " dims, the encoder wants " + std::to_string(feat_dim_));
  }

  // Predictor: [1,T,D] -> alphas [1,T+1] + hidden [1,T+1,D].
  if (pred_.numInputs() != 1 || !shapeIs(pred_.inputShape(0), {1, frames_, hid_dim_})) {
    throw Error(-1, std::string(who) + "predictor input must be " +
                        detail::shapeStr({1, frames_, hid_dim_}));
  }
  pred_alpha_ = pred_hidden_ = -1;
  for (int i = 0; i < pred_.numOutputs(); ++i) {
    const std::vector<int> s = pred_.outputShape(i);
    if (shapeIs(s, {1, frames_ + 1})) pred_alpha_ = i;
    if (shapeIs(s, {1, frames_ + 1, hid_dim_})) pred_hidden_ = i;
  }
  if (pred_alpha_ < 0 || pred_hidden_ < 0) {
    throw Error(-1, std::string(who) + "predictor must output alphas " +
                        detail::shapeStr({1, frames_ + 1}) + " and hidden " +
                        detail::shapeStr({1, frames_ + 1, hid_dim_}));
  }

  // Decoder: four inputs told apart by shape.
  dec_ctx_ = dec_count_ = dec_bias_ = dec_acoustic_ = -1;
  for (int i = 0; i < dec_.numInputs(); ++i) {
    const std::vector<int> s = dec_.inputShape(i);
    if (shapeIs(s, {1, frames_, hid_dim_})) dec_ctx_ = i;
    else if (s.size() == 1 && s[0] == 1) dec_count_ = i;
    else if (shapeIs(s, {1, 1, hid_dim_})) dec_bias_ = i;
    else if (s.size() == 3 && s[0] == 1 && s[2] == hid_dim_) dec_acoustic_ = i;
  }
  if (dec_.numInputs() != 4 || dec_ctx_ < 0 || dec_count_ < 0 || dec_bias_ < 0 ||
      dec_acoustic_ < 0) {
    throw Error(-1, std::string(who) +
                        "decoder must take context [1,T,D], token count [1], bias [1,1,D] "
                        "and acoustic embeddings [1,N,D]");
  }
  max_tokens_ = dec_.inputShape(dec_acoustic_)[1];
  const std::vector<int> lo = dec_.outputShape(0);
  if (lo.size() != 3 || lo[0] != 1 || lo[1] != max_tokens_) {
    throw Error(-1, std::string(who) + "decoder logits must be [1," +
                        std::to_string(max_tokens_) + ",V], got " + detail::shapeStr(lo));
  }
  if (lo[2] != static_cast<int>(vocab_.size())) {
    throw Error(-1, std::string(who) + "decoder has " + std::to_string(lo[2]) +
                        " classes but the token list has " + std::to_string(vocab_.size()) +
                        " — wrong tokens file for this model");
  }
  cfg_.cif.max_tokens = std::min(cfg_.cif.max_tokens, max_tokens_);
  bias_.assign(hid_dim_, 0.0f);
}

float ParaformerAsr::maxSeconds() const noexcept {
  // Each LFR frame advances lfr_n fbank hops.
  return static_cast<float>(frames_) * cfg_.lfr_n * cfg_.fbank.frame_shift_ms / 1000.0f;
}

int ParaformerAsr::features(const float* pcm, std::size_t samples, std::vector<float>& out) {
  const int nfb = fbank_.compute(pcm, samples, fb_);
  const int n = lfrStack(fb_.data(), nfb, cfg_.fbank.num_mel_bins, cfg_.lfr_m, cfg_.lfr_n, out);
  cmvn_.apply(out.data(), n);
  return n;
}

AsrResult ParaformerAsr::transcribe(const float* pcm, std::size_t samples, int timeout_ms) {
  // Only the first frames_ LFR frames reach the encoder. Fbank is per frame,
  // so cut the PCM to the fbank frames those need - one past frames_ * lfr_n,
  // which also keeps the truncation visible - and leave a long recording's
  // tail uncomputed.
  const int lfr_n = cfg_.lfr_n, left = (cfg_.lfr_m - 1) / 2;
  const int need = std::max(frames_ * lfr_n + 1, (frames_ - 1) * lfr_n + cfg_.lfr_m - left);
  const std::size_t cap = static_cast<std::size_t>(fbank_.frameLength()) +
                          static_cast<std::size_t>(need - 1) * fbank_.frameShift();
  const int n = features(pcm, std::min(samples, cap), lfr_);
  AsrResult r = transcribeFeatures(lfr_.data(), n, timeout_ms);
  const int nfb = fbank_.numFrames(samples);
  r.feature_frames = (nfb + lfr_n - 1) / lfr_n;  // what the whole recording would give
  return r;
}

AsrResult ParaformerAsr::transcribeFeatures(const float* feats, int frames, int timeout_ms) {
  AsrResult r;
  r.feature_frames = frames;
  if (frames <= 0) return r;  // shorter than one fbank frame: nothing to say
  if (feats == nullptr) throw Error(-1, "BCDL ParaformerAsr: null features");
  const int valid = std::min(frames, frames_);
  r.truncated = frames > frames_;

  // Encoder input: the utterance, zero-padded to the fixed window.
  x_.assign(static_cast<std::size_t>(frames_) * feat_dim_, 0.0f);
  std::copy_n(feats, static_cast<std::size_t>(valid) * feat_dim_, x_.data());
  setInputFromFloat(enc_, 0, x_.data(), x_.size(), stage_);
  enc_.infer(timeout_ms);
  std::vector<int> shape;
  std::vector<float> scratch;
  const float* ctx = outputAsFloat(enc_, 0, scratch, shape);
  ctx_.assign(ctx, ctx + static_cast<std::size_t>(frames_) * hid_dim_);

  setInputFromFloat(pred_, 0, ctx_.data(), ctx_.size(), stage_);
  pred_.infer(timeout_ms);
  const float* al = outputAsFloat(pred_, pred_alpha_, scratch, shape);
  alphas_.assign(al, al + frames_ + 1);
  std::vector<float> scratch2;
  const float* hd = outputAsFloat(pred_, pred_hidden_, scratch2, shape);
  hidden_.assign(hd, hd + static_cast<std::size_t>(frames_ + 1) * hid_dim_);

  // cfg_.cif.max_tokens is already capped at the decoder's capacity; the
  // decoder input itself is always the full [max_tokens_, D], zero past count.
  const int count = cifIntegrate(alphas_.data(), hidden_.data(), frames_ + 1, hid_dim_, valid,
                                 cfg_.cif, embeds_);
  if (count == 0) return r;
  embeds_.resize(static_cast<std::size_t>(max_tokens_) * hid_dim_, 0.0f);

  const float cnt = static_cast<float>(count);
  setInputFromFloat(dec_, dec_ctx_, ctx_.data(), ctx_.size(), stage_);
  setInputFromFloat(dec_, dec_count_, &cnt, 1, stage_);
  setInputFromFloat(dec_, dec_bias_, bias_.data(), bias_.size(), stage_);
  setInputFromFloat(dec_, dec_acoustic_, embeds_.data(), embeds_.size(), stage_);
  dec_.infer(timeout_ms);
  const float* logits = outputAsFloat(dec_, 0, scratch, shape);
  r.token_ids = argmaxTokens(logits, count, static_cast<int>(vocab_.size()));
  r.text = tokensToText(r.token_ids, vocab_);
  return r;
}

}  // namespace bcdl
