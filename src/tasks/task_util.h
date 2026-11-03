#pragma once

// Internal helpers shared by the speech tasks. Not installed.

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "bcdl/core/status.h"

namespace bcdl {
namespace detail {

/// "[1,80,512]" — for error messages.
inline std::string shapeStr(const std::vector<int>& s) {
  std::string r = "[";
  for (std::size_t i = 0; i < s.size(); ++i) r += (i ? "," : "") + std::to_string(s[i]);
  return r + "]";
}

/// A little-endian float32, C-order, 2-D .npy file. `who` prefixes errors.
inline std::vector<float> loadNpy2d(const std::string& path, int& rows, int& cols,
                                    const std::string& who) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error(-1, "BCDL " + who + ": cannot open " + path);
  char magic[6];
  f.read(magic, 6);
  if (!f || std::string(magic, 6) != "\x93NUMPY")
    throw Error(-1, "BCDL " + who + ": " + path + " is not a .npy file");
  unsigned char ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  std::uint32_t hlen = 0;
  if (ver[0] == 1) {
    unsigned char b[2];
    f.read(reinterpret_cast<char*>(b), 2);
    hlen = b[0] | (b[1] << 8);
  } else {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    hlen = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
  }
  std::string h(hlen, '\0');
  f.read(&h[0], hlen);
  if (h.find("'descr': '<f4'") == std::string::npos ||
      h.find("'fortran_order': False") == std::string::npos)
    throw Error(-1, "BCDL " + who + ": " + path + " must be little-endian float32, C order");
  const auto lp = h.find('(', h.find("'shape'"));
  const auto rp = h.find(')', lp);
  if (lp == std::string::npos || rp == std::string::npos)
    throw Error(-1, "BCDL " + who + ": no shape in " + path);
  std::istringstream ss(h.substr(lp + 1, rp - lp - 1));
  std::vector<long> dims;
  for (std::string tok; std::getline(ss, tok, ',');)
    if (tok.find_first_not_of(" ") != std::string::npos) dims.push_back(std::stol(tok));
  if (dims.size() != 2 || dims[0] <= 0 || dims[1] <= 0)
    throw Error(-1, "BCDL " + who + ": " + path + " must be a 2-D array");
  rows = static_cast<int>(dims[0]);
  cols = static_cast<int>(dims[1]);
  std::vector<float> data(static_cast<std::size_t>(rows) * cols);
  f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * 4));
  if (!f) throw Error(-1, "BCDL " + who + ": " + path + " is truncated");
  return data;
}

}  // namespace detail
}  // namespace bcdl
