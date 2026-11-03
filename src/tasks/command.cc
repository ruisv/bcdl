#include "bcdl/tasks/command.h"

#include <algorithm>
#include <cstdlib>

#include "bcdl/core/status.h"
#include "cn_number.h"

namespace bcdl {

namespace {

#include "pinyin_table.inc"

/// Reading of one CJK code point, or nullptr.
const char* syllableOf(char32_t cp) {
  unsigned short id = 0;
  if (cp >= kPinyinFirst && cp <= kPinyinLast) {
    id = kPinyinIndex[cp - kPinyinFirst];
  } else {
    for (const auto& e : kPinyinExtra)
      if (e.cp == cp) id = e.id;
  }
  return id ? kPinyinSyllables[id] : nullptr;
}

/// Decode one UTF-8 code point at s[i], advancing i. Malformed bytes decode
/// to 0xFFFD one byte at a time, so garbage is skipped rather than fatal.
char32_t nextCodePoint(const std::string& s, std::size_t& i) {
  auto b = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
  const unsigned char c = b(i);
  int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
  if (len == 0 || i + len > s.size()) {
    ++i;
    return 0xFFFD;
  }
  char32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
  for (int k = 1; k < len; ++k) {
    if ((b(i + k) & 0xC0) != 0x80) {
      ++i;
      return 0xFFFD;
    }
    cp = (cp << 6) | (b(i + k) & 0x3F);
  }
  i += len;
  return cp;
}

/// Full-width ASCII (U+FF01..U+FF5E) -> ASCII.
char32_t foldWidth(char32_t cp) { return (cp >= 0xFF01 && cp <= 0xFF5E) ? cp - 0xFEE0 : cp; }

bool isDigit(char32_t c) { return c >= '0' && c <= '9'; }
bool isLetter(char32_t c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

/// Read a digit run the way it is spoken, as pinyin: 90 -> jiu shi.
void readNumber(const std::string& digits, std::vector<std::string>& out) {
  const std::string hanzi = detail::cnNumber(digits);
  for (std::size_t i = 0; i < hanzi.size();) {
    if (const char* s = syllableOf(nextCodePoint(hanzi, i))) out.emplace_back(s);
  }
}

/// toPinyin(), optionally tagging Latin words so they can never equal a
/// pinyin syllable: the recogniser writing "you" means it heard English, not
/// 右 (a real miss turned 掉头 into "you too", which then matched 右转).
std::vector<std::string> tokenize(const std::string& text, bool tag_words) {
  std::vector<std::string> out;
  std::string run;  // pending ASCII letter or digit run
  bool run_digits = false;
  auto flush = [&] {
    if (run.empty()) return;
    if (run_digits) {
      readNumber(run, out);
    } else {
      out.push_back(tag_words ? "#" + run : run);
    }
    run.clear();
  };
  for (std::size_t i = 0; i < text.size();) {
    const char32_t cp = foldWidth(nextCodePoint(text, i));
    if (isLetter(cp) || isDigit(cp)) {
      const bool d = isDigit(cp);
      if (!run.empty() && d != run_digits) flush();
      run_digits = d;
      run += static_cast<char>(isLetter(cp) ? (cp | 0x20) : cp);
      continue;
    }
    flush();
    if (const char* s = syllableOf(cp)) out.emplace_back(s);
  }
  flush();
  return out;
}

}  // namespace

std::vector<std::string> toPinyin(const std::string& text) { return tokenize(text, false); }

CommandMatcher::CommandMatcher(std::vector<std::string> commands, float max_distance)
    : commands_(std::move(commands)), max_distance_(max_distance) {
  if (commands_.empty()) throw Error(-1, "BCDL CommandMatcher: the command list is empty");
  if (!(max_distance_ >= 0.0f))
    throw Error(-1, "BCDL CommandMatcher: max_distance must be >= 0");
  encoded_.reserve(commands_.size());
  for (const auto& c : commands_) {
    std::vector<int> e;
    for (auto& tok : tokenize(c, true)) {
      auto it = ids_.emplace(std::move(tok), static_cast<int>(ids_.size())).first;
      e.push_back(it->second);
    }
    if (e.empty())
      throw Error(-1, "BCDL CommandMatcher: command '" + c +
                          "' has no pinyin (no CJK, Latin letters or digits)");
    encoded_.push_back(std::move(e));
  }
}

std::vector<int> CommandMatcher::encode(const std::string& text) const {
  std::vector<int> e;
  for (const auto& tok : tokenize(text, true)) {
    auto it = ids_.find(tok);
    e.push_back(it == ids_.end() ? -1 : it->second);  // -1 never equals a command token
  }
  return e;
}

namespace {

int editDistance(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& row) {
  row.resize(b.size() + 1);
  for (std::size_t j = 0; j <= b.size(); ++j) row[j] = static_cast<int>(j);
  for (std::size_t i = 1; i <= a.size(); ++i) {
    int diag = row[0];
    row[0] = static_cast<int>(i);
    for (std::size_t j = 1; j <= b.size(); ++j) {
      const int up = row[j];
      row[j] = std::min({row[j] + 1, row[j - 1] + 1, diag + (a[i - 1] != b[j - 1])});
      diag = up;
    }
  }
  return row[b.size()];
}

}  // namespace

std::vector<float> CommandMatcher::distances(const std::string& text) const {
  const auto t = encode(text);
  std::vector<float> d(commands_.size(), 1e30f);
  if (t.empty()) return d;
  std::vector<int> row;
  for (std::size_t k = 0; k < encoded_.size(); ++k)
    d[k] = static_cast<float>(editDistance(t, encoded_[k], row)) /
           static_cast<float>(encoded_[k].size());
  return d;
}

CommandMatch CommandMatcher::match(const std::string& text) const {
  CommandMatch m;
  const auto t = encode(text);
  if (t.empty()) return m;
  std::vector<int> row;
  long best_len_gap = 0;
  for (std::size_t k = 0; k < encoded_.size(); ++k) {
    const float d = static_cast<float>(editDistance(t, encoded_[k], row)) /
                    static_cast<float>(encoded_[k].size());
    const long gap = std::labs(static_cast<long>(t.size()) - static_cast<long>(encoded_[k].size()));
    if (m.nearest < 0 || d < m.distance || (d == m.distance && gap < best_len_gap)) {
      m.nearest = static_cast<int>(k);
      m.distance = d;
      best_len_gap = gap;
    }
  }
  if (m.distance <= max_distance_) {
    m.index = m.nearest;
    m.command = commands_[m.index];
  }
  return m;
}

}  // namespace bcdl
