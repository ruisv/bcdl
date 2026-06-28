#include "bcdl/core/sys_mem.h"

#include <utility>

#include "bcdl/core/status.h"

namespace bcdl {

SysMem::SysMem(uint64_t size, bool cached, int device_id) : cached_(cached) {
  if (cached) {
    BCDL_CHECK(hbUCPMallocCached(&mem_, size, device_id));
  } else {
    BCDL_CHECK(hbUCPMalloc(&mem_, size, device_id));
  }
}

SysMem::~SysMem() { release(); }

SysMem::SysMem(SysMem&& other) noexcept : mem_(other.mem_), cached_(other.cached_) {
  other.mem_ = hbUCPSysMem{};
}

SysMem& SysMem::operator=(SysMem&& other) noexcept {
  if (this != &other) {
    release();
    mem_ = other.mem_;
    cached_ = other.cached_;
    other.mem_ = hbUCPSysMem{};
  }
  return *this;
}

void SysMem::cleanCache() const {
  if (cached_ && mem_.virAddr) {
    BCDL_CHECK(hbUCPMemFlush(&mem_, HB_SYS_MEM_CACHE_CLEAN));
  }
}

void SysMem::invalidateCache() const {
  if (cached_ && mem_.virAddr) {
    BCDL_CHECK(hbUCPMemFlush(&mem_, HB_SYS_MEM_CACHE_INVALIDATE));
  }
}

void SysMem::release() noexcept {
  if (mem_.virAddr) {
    // best-effort free; never throw from a destructor
    hbUCPFree(&mem_);
    mem_ = hbUCPSysMem{};
  }
}

}  // namespace bcdl
