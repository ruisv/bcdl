#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "bcdl/core/sys_mem.h"

namespace bcdl {

/// A reusable pool of cached SysMem blocks for zero-per-frame-allocation
/// pipelines.
///
/// Every S100 subsystem (BPU tensors, JPU/VPU codecs, VP preprocessing) is
/// unified on `bcdl::SysMem` (RAII over `hbUCPSysMem`). Allocating a fresh
/// buffer on every frame means a `hbUCPMallocCached` per frame, which is
/// wasteful for a streaming pipeline. MemPool pre-allocates a set of blocks and
/// hands them out / recycles them so a steady-state pipeline performs no device
/// allocation at all.
///
/// Blocks are served by capacity: `acquire(size)` returns the smallest free
/// block whose capacity is >= `size`. If no free block fits, a new one is
/// allocated, rounding the request up to `kGranularity` (256 bytes) to improve
/// the chance that a later, slightly different request can reuse it. Blocks are
/// handed back through an RAII `Lease`; when the Lease is destroyed (or moved
/// from / reset by being overwritten) the block returns to the free set.
///
/// Cache discipline is unchanged from SysMem: the block is cached, so after the
/// CPU writes an input call `lease.cleanCache()` before handing it to the
/// device, and after the device writes call `lease.invalidateCache()` before
/// the CPU reads. MemPool does NOT clear or zero recycled buffers — a reused
/// block still holds the previous tenant's bytes, which is fine because callers
/// always overwrite the logical region before use.
///
/// Thread-safety: acquire/reserve/clear and Lease return are all guarded by an
/// internal mutex, so a producer thread can acquire while a consumer thread
/// releases. The buffers themselves carry no synchronization; that is the
/// caller's responsibility (as with any SysMem).
///
/// Lifetime contract: a MemPool must outlive every Lease it has handed out. A
/// Lease holds a back-reference into the pool and dereferences it on
/// destruction; destroying the pool while a Lease is still live is undefined.
/// This mirrors typical pool usage (pool owned by the pipeline, leases scoped to
/// a frame).
class MemPool {
 public:
  /// Allocation granularity: requested sizes are rounded up to this when a new
  /// block must be allocated, so near-equal requests share blocks.
  static constexpr uint64_t kGranularity = 256;

  /// `cached`/`device_id` are forwarded to every SysMem this pool allocates.
  explicit MemPool(bool cached = true, int device_id = 0);
  ~MemPool();

  MemPool(const MemPool&) = delete;
  MemPool& operator=(const MemPool&) = delete;

 private:
  /// One owned buffer plus its bookkeeping. Heap-allocated and held by
  /// unique_ptr so its address is stable across pool growth — a Lease references
  /// it by raw pointer.
  struct Block {
    SysMem mem;
    uint64_t capacity = 0;  // == mem.size(); cached for locking-free reads
    bool in_use = false;
  };

 public:
  /// Move-only RAII handle to a leased block. On destruction it returns the
  /// block to the owning pool's free set. The block's capacity may exceed the
  /// requested size; `size()` reports the logical size requested at acquire().
  class Lease {
   public:
    Lease() = default;
    ~Lease();

    Lease(Lease&& other) noexcept;
    Lease& operator=(Lease&& other) noexcept;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;

    /// The underlying buffer. Precondition: valid().
    SysMem& mem() const;
    /// Convenience: mem().data().
    void* data() const;
    /// Allocated capacity of the block (>= size()).
    uint64_t capacity() const;
    /// Logical size requested at acquire() (<= capacity()).
    uint64_t size() const { return size_; }
    /// True if this Lease owns a block (false after move-from / default ctor).
    bool valid() const noexcept { return pool_ != nullptr && block_ != nullptr; }

    /// Flush CPU writes to DRAM (forward to mem().cleanCache()).
    void cleanCache() const;
    /// Drop stale CPU cache lines (forward to mem().invalidateCache()).
    void invalidateCache() const;

   private:
    friend class MemPool;
    Lease(MemPool* pool, Block* block, uint64_t size)
        : pool_(pool), block_(block), size_(size) {}
    void reset() noexcept;

    MemPool* pool_ = nullptr;
    Block* block_ = nullptr;
    uint64_t size_ = 0;
  };

  /// Return a block with capacity >= `size`, reusing a free one if possible or
  /// allocating a new block (rounded up to kGranularity) otherwise. Thread-safe.
  /// A size of 0 yields a valid Lease over a (possibly minimum-rounded) block.
  Lease acquire(uint64_t size);

  /// Pre-allocate `count` blocks each with capacity >= `size` (rounded up to
  /// kGranularity), added to the free set. Use at startup to warm the pool so
  /// steady-state acquire() never allocates. Thread-safe.
  void reserve(uint64_t size, int count);

  /// Total blocks owned (free + leased).
  std::size_t blockCount() const;
  /// Blocks currently free (available to acquire without allocation).
  std::size_t freeCount() const;
  /// Sum of all owned block capacities, in bytes.
  uint64_t bytesPooled() const;

  /// Free every block that is NOT currently leased. Leased blocks are left
  /// untouched (their Leases still own them and will return them to a then-empty
  /// free set). Thread-safe.
  void clear();

 private:
  /// Called by Lease::~Lease / reset to return a block. Marks it free.
  void release(Block* block) noexcept;

  static uint64_t roundUp(uint64_t size) noexcept {
    return ((size + kGranularity - 1) / kGranularity) * kGranularity;
  }

  mutable std::mutex mu_;
  std::vector<std::unique_ptr<Block>> blocks_;
  bool cached_;
  int device_id_;
};

}  // namespace bcdl
