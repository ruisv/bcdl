#include "bcdl/core/task.h"

#include "bcdl/core/status.h"

namespace bcdl {

void Task::submit(int priority) {
  hbUCPSchedParam sched;
  HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
  sched.priority = priority;
  BCDL_CHECK(hbUCPSubmitTask(handle_, &sched));
}

void Task::wait(int timeout_ms) { BCDL_CHECK(hbUCPWaitTaskDone(handle_, timeout_ms)); }

void Task::release() noexcept {
  if (handle_) {
    hbUCPReleaseTask(handle_);
    handle_ = nullptr;
  }
}

}  // namespace bcdl
