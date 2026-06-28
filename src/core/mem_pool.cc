#include "bcdl/core/mem_pool.h"

#include <algorithm>
#include <utility>

namespace bcdl {

// ---------------------------------------------------------------------------
// MemPool
// ---------------------------------------------------------------------------

MemPool::MemPool(bool cached, int device_id) : cached_(cached), device_id_(device_id) {}

MemPool::~MemPool() = default;

MemPool::Lease MemPool::acquire(uint64_t size) {
  std::lock_guard<std::mutex> lock(mu_);

  // Find the smallest free block that fits, to avoid handing out an oversized
  // block when a tighter one is available (best-fit by capacity).
  Block* best = nullptr;
  for (const auto& b : blocks_) {
    if (b->in_use || b->capacity < size) continue;
    if (best == nullptr || b->capacity < best->capacity) best = b.get();
  }

  if (best == nullptr) {
    // Nothing fits: allocate a new block rounded up to the granularity (and at
    // least one granule, so a zero-size request still yields a real buffer).
    const uint64_t cap = std::max(roundUp(size), kGranularity);
    auto block = std::make_unique<Block>();
    block->mem = SysMem(cap, cached_, device_id_);
    block->capacity = block->mem.size();  // SysMem may report >= requested
    block->in_use = true;
    best = block.get();
    blocks_.push_back(std::move(block));
  } else {
    best->in_use = true;
  }

  return Lease(this, best, size);
}

void MemPool::reserve(uint64_t size, int count) {
  if (count <= 0) return;
  const uint64_t cap = std::max(roundUp(size), kGranularity);

  std::lock_guard<std::mutex> lock(mu_);
  blocks_.reserve(blocks_.size() + static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    auto block = std::make_unique<Block>();
    block->mem = SysMem(cap, cached_, device_id_);
    block->capacity = block->mem.size();
    block->in_use = false;
    blocks_.push_back(std::move(block));
  }
}

void MemPool::release(Block* block) noexcept {
  if (block == nullptr) return;
  std::lock_guard<std::mutex> lock(mu_);
  block->in_use = false;
}

std::size_t MemPool::blockCount() const {
  std::lock_guard<std::mutex> lock(mu_);
  return blocks_.size();
}

std::size_t MemPool::freeCount() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::size_t n = 0;
  for (const auto& b : blocks_) {
    if (!b->in_use) ++n;
  }
  return n;
}

uint64_t MemPool::bytesPooled() const {
  std::lock_guard<std::mutex> lock(mu_);
  uint64_t total = 0;
  for (const auto& b : blocks_) total += b->capacity;
  return total;
}

void MemPool::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  // Drop only the free blocks; leased blocks remain owned and will return to the
  // (now smaller) free set when their Leases are destroyed.
  blocks_.erase(std::remove_if(blocks_.begin(), blocks_.end(),
                               [](const std::unique_ptr<Block>& b) { return !b->in_use; }),
                blocks_.end());
}

// ---------------------------------------------------------------------------
// MemPool::Lease
// ---------------------------------------------------------------------------

MemPool::Lease::~Lease() { reset(); }

MemPool::Lease::Lease(Lease&& other) noexcept
    : pool_(other.pool_), block_(other.block_), size_(other.size_) {
  other.pool_ = nullptr;
  other.block_ = nullptr;
  other.size_ = 0;
}

MemPool::Lease& MemPool::Lease::operator=(Lease&& other) noexcept {
  if (this != &other) {
    reset();
    pool_ = other.pool_;
    block_ = other.block_;
    size_ = other.size_;
    other.pool_ = nullptr;
    other.block_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void MemPool::Lease::reset() noexcept {
  if (pool_ != nullptr && block_ != nullptr) {
    pool_->release(block_);
  }
  pool_ = nullptr;
  block_ = nullptr;
  size_ = 0;
}

SysMem& MemPool::Lease::mem() const { return block_->mem; }

void* MemPool::Lease::data() const { return block_->mem.data(); }

uint64_t MemPool::Lease::capacity() const { return block_->capacity; }

void MemPool::Lease::cleanCache() const { block_->mem.cleanCache(); }

void MemPool::Lease::invalidateCache() const { block_->mem.invalidateCache(); }

}  // namespace bcdl
