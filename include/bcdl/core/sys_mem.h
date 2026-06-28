#pragma once

#include <cstdint>

#include "hobot/hb_ucp_sys.h"

namespace bcdl {

/// RAII wrapper over an hbUCPSysMem block — the single shared-memory buffer type
/// used by BPU tensors, JPU/VPU codecs and VP preprocessing on S100.
///
/// Allocated cached by default; the caller is responsible for cache coherency
/// across CPU<->device hand-offs:
///   - CPU wrote, device will read  -> cleanCache()      (CACHE_CLEAN)
///   - device wrote, CPU will read  -> invalidateCache() (CACHE_INVALIDATE)
class SysMem {
 public:
  SysMem() = default;
  /// Allocate `size` bytes. cached=true uses hbUCPMallocCached (needs explicit
  /// flush/invalidate); cached=false uses hbUCPMalloc (coherent, slower CPU access).
  explicit SysMem(uint64_t size, bool cached = true, int device_id = 0);
  ~SysMem();

  SysMem(const SysMem&) = delete;
  SysMem& operator=(const SysMem&) = delete;
  SysMem(SysMem&& other) noexcept;
  SysMem& operator=(SysMem&& other) noexcept;

  /// Flush CPU cache to DRAM so the device sees CPU writes.
  void cleanCache() const;
  /// Drop stale CPU cache lines so the CPU re-reads device writes.
  void invalidateCache() const;

  void* data() const noexcept { return mem_.virAddr; }
  uint64_t phyAddr() const noexcept { return mem_.phyAddr; }
  uint64_t size() const noexcept { return mem_.memSize; }
  bool valid() const noexcept { return mem_.virAddr != nullptr; }

  /// Raw struct, e.g. to assign into hbDNNTensor.sysMem (shallow copy of addrs).
  hbUCPSysMem& raw() noexcept { return mem_; }
  const hbUCPSysMem& raw() const noexcept { return mem_; }

 private:
  void release() noexcept;

  hbUCPSysMem mem_{};
  bool cached_ = true;
};

}  // namespace bcdl
