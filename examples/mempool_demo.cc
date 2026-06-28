// M4 demo: exercises the SysMem MemPool — pre-allocate + reuse, no .hbm needed.
//
//   ./mempool_demo
//
// Reserves a few blocks up front, then runs an acquire/release loop and prints
// blockCount/freeCount/bytesPooled to show that steady-state acquires reuse
// existing blocks (blockCount stays flat once warm) rather than allocating.
// Includes the core header directly (MemPool is not yet re-exported via bcdl.h).

#include <cstdint>
#include <cstdio>

#include "bcdl/core/mem_pool.h"

static void dump(const char* tag, const bcdl::MemPool& pool) {
  std::printf("%-22s blocks=%zu free=%zu pooled=%llu B\n", tag, pool.blockCount(),
              pool.freeCount(), static_cast<unsigned long long>(pool.bytesPooled()));
}

int main() {
  try {
    bcdl::MemPool pool(/*cached=*/true, /*device_id=*/0);

    // Warm the pool with 4 blocks of >= 1 MiB each.
    const uint64_t kBuf = 1u << 20;
    pool.reserve(kBuf, 4);
    dump("after reserve(1MiB,4)", pool);

    // Streaming loop: acquire a frame buffer, "use" it, release at scope end.
    for (int frame = 0; frame < 8; ++frame) {
      bcdl::MemPool::Lease lease = pool.acquire(kBuf);
      // Touch the buffer the way a real producer would: write then flush.
      if (lease.valid()) {
        auto* p = static_cast<unsigned char*>(lease.data());
        p[0] = static_cast<unsigned char>(frame);
        lease.cleanCache();
      }
      std::printf("frame %d: cap=%llu size=%llu free-now=%zu\n", frame,
                  static_cast<unsigned long long>(lease.capacity()),
                  static_cast<unsigned long long>(lease.size()), pool.freeCount());
      // lease destroyed here -> block returns to the pool.
    }
    dump("after loop", pool);

    // Hold two leases at once to force one growth beyond the reserved 4.
    {
      bcdl::MemPool::Lease a = pool.acquire(kBuf);
      bcdl::MemPool::Lease b = pool.acquire(kBuf);
      bcdl::MemPool::Lease c = pool.acquire(kBuf);
      bcdl::MemPool::Lease d = pool.acquire(kBuf);
      bcdl::MemPool::Lease e = pool.acquire(kBuf);  // 5th concurrent -> grows
      dump("5 concurrent leases", pool);
      (void)a;
      (void)b;
      (void)c;
      (void)d;
      (void)e;
    }
    dump("after release", pool);

    // A larger request cannot reuse the 1 MiB blocks -> allocates a bigger one.
    {
      bcdl::MemPool::Lease big = pool.acquire(4u << 20);
      std::printf("big: cap=%llu (rounded from %u)\n",
                  static_cast<unsigned long long>(big.capacity()), 4u << 20);
    }
    dump("after big", pool);

    pool.clear();  // frees all free blocks (none leased here)
    dump("after clear", pool);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mempool_demo error: %s\n", e.what());
    return 1;
  }
}
