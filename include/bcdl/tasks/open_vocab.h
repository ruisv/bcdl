#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bcdl {

// ===========================================================================
// Open-vocabulary detection label table (YOLOE / YOLOE-26, prompt-free export)
// ===========================================================================
//
// In prompt-free / preset-vocabulary mode a YOLOE model's class axis is a
// FIXED vocabulary baked into the head at export time (the CLIP text embeddings
// are folded into the 1x1 classification conv). At runtime the detector is a
// plain anchor-free head — decode is UNCHANGED (reuse Detector /
// YoloLtrbDetector). The only new runtime state is the class_id -> prompt-name
// map, which this holds. Keep the `.hbm` and its `labels.txt` (one prompt per
// line) side by side; `num_classes` in DetectConfig/YoloLtrbConfig must equal
// LabelMap::size().

/// A class_id -> label-name table for an open-vocabulary detection head.
struct LabelMap {
  std::vector<std::string> names;

  /// Load one label per line (leading/trailing whitespace trimmed; blank lines
  /// dropped). Throws std::runtime_error if the file cannot be opened.
  static LabelMap fromFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
      throw std::runtime_error("BCDL LabelMap: cannot open labels file: " + path);
    }
    LabelMap m;
    std::string line;
    while (std::getline(f, line)) {
      // Trim CR (Windows line endings) + surrounding whitespace.
      const char* ws = " \t\r\n";
      const size_t b = line.find_first_not_of(ws);
      if (b == std::string::npos) continue;  // blank line
      const size_t e = line.find_last_not_of(ws);
      m.names.push_back(line.substr(b, e - b + 1));
    }
    return m;
  }

  /// Build from an in-memory list (e.g. a custom prompt vocabulary).
  static LabelMap fromList(std::vector<std::string> v) {
    LabelMap m;
    m.names = std::move(v);
    return m;
  }

  size_t size() const { return names.size(); }

  /// Name for a class id, or "?" when out of range (never throws — decode may
  /// legitimately emit ids beyond a truncated table during debugging).
  const std::string& name(int id) const {
    static const std::string kUnknown = "?";
    if (id < 0 || id >= static_cast<int>(names.size())) return kUnknown;
    return names[static_cast<size_t>(id)];
  }
};

}  // namespace bcdl
