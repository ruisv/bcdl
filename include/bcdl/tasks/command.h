#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace bcdl {

// ===========================================================================
// Closed-set voice commands: recogniser text -> one of N known commands
// ===========================================================================
//
// A robot that takes spoken commands does not need a perfect transcript; it
// needs to know WHICH command was said. Comparing in toneless pinyin instead
// of characters absorbs the recogniser's most common mistakes on short
// Mandarin commands — homophones (打开灯 -> 大开灯, 出发 -> 处罚, 关灯 -> 灌灯)
// and a stray trailing particle (前进 -> 前进了) — and a distance threshold
// rejects speech that is not a command at all.
//
// On a 40-command set read by 9 synthetic voices, Paraformer's exact-sentence
// accuracy was 84.7% clean / 82.8% with 5 dB noise; matched this way it was
// 96.9% / 95.6%, with 0.6% / 1.1% mapped to the wrong command and none of 600
// news sentences accepted as a command. Pure CPU, microseconds per call.

/// Toneless pinyin of `text` (UTF-8), one token per syllable.
///
/// - CJK characters (U+3400..U+9FFF, 〇) become their most common reading,
///   tone dropped, ü written `v`. Polyphones are not disambiguated by
///   context; that does not matter for matching, because commands and
///   recogniser output go through the same table.
/// - A run of ASCII letters becomes one lower-cased token ("OK" -> "ok").
/// - A run of digits is read as a Chinese number first ("90" -> jiu shi,
///   "105" -> yi bai ling wu), so "转90度" matches "转九十度". Runs longer than
///   8 digits or with a leading zero are read digit by digit.
/// - Full-width letters and digits count as ASCII; everything else
///   (punctuation, spaces, other scripts) is dropped.
std::vector<std::string> toPinyin(const std::string& text);

/// Result of one match. `index` is the accepted command, or -1 when the
/// nearest one was still too far (or the text had no pinyin at all).
struct CommandMatch {
  int index = -1;          ///< accepted command, -1 = rejected
  int nearest = -1;        ///< closest command even when rejected (-1 only for empty text)
  float distance = 1e30f;  ///< syllable edit distance / syllables in that command
  std::string command;     ///< commands()[index], empty when rejected
};

/// Matches recogniser output against a fixed list of commands.
///
/// Distance is the syllable-level Levenshtein distance divided by the
/// command's syllable count, so 0 is an exact (homophone) match and 1 means
/// as many edits as the command is long. Ties go to the command whose length
/// is closer to the text's, then to the earlier one. A match is accepted when
/// the distance is <= `max_distance`; 0.5 (half the syllables may be wrong)
/// was the setting measured above. Lower it for long commands that differ by
/// one syllable; raise it only with a short, very distinct list.
///
/// Latin words are compared with Latin words only, never with a syllable that
/// happens to be spelled the same ("you" is not 右).
///
/// Several phrasings of one intent are just several entries; map indices to
/// intents yourself.
class CommandMatcher {
 public:
  explicit CommandMatcher(std::vector<std::string> commands, float max_distance = 0.5f);

  CommandMatch match(const std::string& text) const;

  /// Distance from `text` to every command, in list order (empty text: all 1e30).
  std::vector<float> distances(const std::string& text) const;

  const std::vector<std::string>& commands() const noexcept { return commands_; }
  int size() const noexcept { return static_cast<int>(commands_.size()); }
  float maxDistance() const noexcept { return max_distance_; }

 private:
  std::vector<int> encode(const std::string& text) const;

  std::vector<std::string> commands_;
  float max_distance_;
  std::unordered_map<std::string, int> ids_;  // token -> id, over all command tokens
  std::vector<std::vector<int>> encoded_;
};

}  // namespace bcdl
