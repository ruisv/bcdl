#include "bcdl/core/status.h"

#include <sstream>

namespace bcdl::detail {

void check(int ret, const char* expr, const char* file, int line) {
  if (ret != 0) {
    std::ostringstream os;
    os << "BCDL: hobot call failed (code " << ret << "): " << expr << "\n  at " << file
       << ":" << line;
    throw Error(ret, os.str());
  }
}

}  // namespace bcdl::detail
