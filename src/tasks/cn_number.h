#pragma once

// Internal: Arabic digits -> how a Mandarin speaker reads them, in characters.
// Shared by the command matcher (which then takes the pinyin) and the TTS
// front end (which looks the characters up in its lexicon). Not installed.

#include <cstdlib>
#include <string>

namespace bcdl {
namespace detail {

inline const char* cnDigit(int d) {
  static const char* const k[] = {"零", "一", "二", "三", "四", "五", "六", "七", "八", "九"};
  return k[d];
}

/// 1..9999. `full` keeps the 一 of 一十 (after a higher unit: 一百一十,
/// 一万零一十); otherwise 10..19 read 十x.
inline void cnBelow10k(int n, bool full, std::string& out) {
  static const char* const unit[] = {"千", "百", "十", ""};
  const int d[4] = {n / 1000, n / 100 % 10, n / 10 % 10, n % 10};
  bool started = false, gap = false;
  for (int k = 0; k < 4; ++k) {
    if (d[k] == 0) {
      gap = started;
      continue;
    }
    if (gap) out += "零";
    gap = false;
    const bool bare_ten = k == 2 && d[k] == 1 && !started && !full;
    if (!bare_ten) out += cnDigit(d[k]);
    out += unit[k];
    started = true;
  }
}

/// Read a run of ASCII digits as a number (cn2an "low" style): 90 -> 九十,
/// 105 -> 一百零五, 10010 -> 一万零一十. A leading zero or more than 8 digits
/// reads digit by digit (codes, phone numbers).
inline std::string cnNumber(const std::string& digits) {
  std::string out;
  if (digits.empty()) return out;
  if (digits.size() > 8 || (digits.size() > 1 && digits[0] == '0')) {
    for (char c : digits) out += cnDigit(c - '0');
    return out;
  }
  const long n = std::strtol(digits.c_str(), nullptr, 10);
  if (n == 0) return "零";
  const int hi = static_cast<int>(n / 10000), lo = static_cast<int>(n % 10000);
  if (hi) {
    cnBelow10k(hi, false, out);
    out += "万";
  }
  if (lo) {
    if (hi && lo < 1000) out += "零";
    cnBelow10k(lo, hi != 0, out);
  }
  return out;
}

/// Each digit on its own: 2026 -> 二零二六 (years, codes).
inline std::string cnDigits(const std::string& digits) {
  std::string out;
  for (char c : digits) out += cnDigit(c - '0');
  return out;
}

}  // namespace detail
}  // namespace bcdl
