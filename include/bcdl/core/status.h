#pragma once

#include <stdexcept>
#include <string>

namespace bcdl {

/// Exception carrying a hobot SDK return code (hbDNN / hbUCP / hbVP all use int32_t,
/// 0 == success).
class Error : public std::runtime_error {
 public:
  Error(int code, const std::string& what) : std::runtime_error(what), code_(code) {}
  int code() const noexcept { return code_; }

 private:
  int code_;
};

namespace detail {
/// Throws bcdl::Error if ret != 0. Used by the BCDL_CHECK macro.
void check(int ret, const char* expr, const char* file, int line);
}  // namespace detail

}  // namespace bcdl

/// Wrap every hobot SDK call: BCDL_CHECK(hbDNNInferV2(...));
#define BCDL_CHECK(expr) ::bcdl::detail::check((expr), #expr, __FILE__, __LINE__)
