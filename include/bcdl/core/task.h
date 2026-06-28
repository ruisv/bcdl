#pragma once

#include "hobot/hb_ucp.h"

namespace bcdl {

/// RAII wrapper over an hbUCPTaskHandle_t — the single task type returned by
/// hbDNNInferV2 and every hbVP codec / preprocessing op. Submit it to the hbUCP
/// scheduler, wait for completion, and release.
///
/// Typical use:
///   Task t;
///   hbDNNInferV2(t.addr(), outputs, inputs, dnn);  // fills the handle
///   t.submit();
///   t.wait();                                       // blocks
class Task {
 public:
  Task() = default;
  explicit Task(hbUCPTaskHandle_t handle) : handle_(handle) {}
  ~Task() { release(); }

  Task(const Task&) = delete;
  Task& operator=(const Task&) = delete;
  Task(Task&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
  Task& operator=(Task&& other) noexcept {
    if (this != &other) {
      release();
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  /// Address to pass to producers that fill the handle (hbDNNInferV2, hbVP*...).
  hbUCPTaskHandle_t* addr() noexcept { return &handle_; }
  hbUCPTaskHandle_t get() const noexcept { return handle_; }
  bool valid() const noexcept { return handle_ != nullptr; }

  /// Enqueue on the hbUCP scheduler with default scheduling params.
  void submit(int priority = HB_UCP_PRIORITY_LOWEST);
  /// Block until done. timeout_ms == 0 blocks indefinitely (UCP convention).
  void wait(int timeout_ms = 0);

  /// Free the handle now (also done by the destructor).
  void release() noexcept;

 private:
  hbUCPTaskHandle_t handle_ = nullptr;
};

}  // namespace bcdl
