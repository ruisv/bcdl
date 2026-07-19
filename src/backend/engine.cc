#include "bcdl/backend/engine.h"

#include <cstring>
#include <stdexcept>

#include "bcdl/core/status.h"
#include "bcdl/core/task.h"
#include "hobot/hb_ucp.h"

namespace bcdl {

namespace {
// BPU width-stride alignment: S100/S100P/S600 -> 32, S300 -> 64.
constexpr int64_t kBpuStrideAlign = 32;
inline int64_t alignUp(int64_t v, int64_t a) { return (v + (a - 1)) & ~(a - 1); }
}  // namespace

std::vector<std::string> Engine::modelNames(const std::string& hbm_path) {
  hbDNNPackedHandle_t packed = nullptr;
  const char* files[] = {hbm_path.c_str()};
  BCDL_CHECK(hbDNNInitializeFromFiles(&packed, files, 1));
  std::vector<std::string> out;
  try {
    const char** names = nullptr;
    int name_count = 0;
    BCDL_CHECK(hbDNNGetModelNameList(&names, &name_count, packed));
    out.reserve(static_cast<std::size_t>(name_count));
    for (int i = 0; i < name_count; ++i) out.emplace_back(names[i]);
  } catch (...) {
    hbDNNRelease(packed);
    throw;
  }
  hbDNNRelease(packed);
  return out;
}

Engine::Engine(const std::string& hbm_path, const std::string& model_name) {
  const char* files[] = {hbm_path.c_str()};
  BCDL_CHECK(hbDNNInitializeFromFiles(&packed_, files, 1));

  // Once packed_ owns a handle, any later throw must release it — the object is
  // never fully constructed, so ~Engine() would not run (handle/device leak).
  try {
    // Resolve which model in the package to run.
    const char** names = nullptr;
    int name_count = 0;
    BCDL_CHECK(hbDNNGetModelNameList(&names, &name_count, packed_));
    if (name_count <= 0) {
      throw Error(-1, "BCDL: hbm package contains no models: " + hbm_path);
    }
    model_name_ = model_name.empty() ? names[0] : model_name;
    BCDL_CHECK(hbDNNGetModelHandle(&dnn_, packed_, model_name_.c_str()));

    allocTensors();
  } catch (...) {
    hbDNNRelease(packed_);
    packed_ = nullptr;
    throw;
  }
}

Engine::~Engine() {
  // SysMem members free their device buffers first (declared after, destroyed before).
  if (packed_) {
    hbDNNRelease(packed_);
    packed_ = nullptr;
  }
}

void Engine::allocTensors() {
  int in_count = 0, out_count = 0;
  BCDL_CHECK(hbDNNGetInputCount(&in_count, dnn_));
  BCDL_CHECK(hbDNNGetOutputCount(&out_count, dnn_));

  inputs_.resize(in_count);
  outputs_.resize(out_count);
  input_mem_.reserve(in_count);
  output_mem_.reserve(out_count);

  for (int i = 0; i < in_count; ++i) {
    auto& p = inputs_[i].properties;
    BCDL_CHECK(hbDNNGetInputTensorProperties(&p, dnn_, i));
    // Input alignedByteSize/stride are dynamic (-1). Resolve byte strides from
    // innermost to outermost (BPU-aligned), then size = stride[0] * dim[0].
    const int nd = p.validShape.numDimensions;
    if (nd > 0 && p.stride[nd - 1] == -1) {
      p.stride[nd - 1] = static_cast<int64_t>(elemSize(p.tensorType));
    }
    for (int d = nd - 2; d >= 0; --d) {
      if (p.stride[d] == -1) {
        p.stride[d] = alignUp(p.stride[d + 1] * p.validShape.dimensionSize[d + 1],
                              kBpuStrideAlign);
      }
    }
    const uint64_t bytes =
        nd > 0 ? static_cast<uint64_t>(p.stride[0] * p.validShape.dimensionSize[0]) : 0;
    input_mem_.emplace_back(bytes, /*cached=*/true);
    inputs_[i].sysMem = input_mem_[i].raw();
  }
  for (int i = 0; i < out_count; ++i) {
    BCDL_CHECK(hbDNNGetOutputTensorProperties(&outputs_[i].properties, dnn_, i));
    output_mem_.emplace_back(static_cast<uint64_t>(outputs_[i].properties.alignedByteSize),
                             /*cached=*/true);
    outputs_[i].sysMem = output_mem_[i].raw();
  }
}

void Engine::setInput(int i, const void* data, std::size_t bytes) {
  if (i < 0 || i >= numInputs()) throw Error(-1, "BCDL: input index out of range");
  if (bytes > input_mem_[i].size()) {
    throw Error(-1, "BCDL: input " + std::to_string(i) + " too large (" +
                        std::to_string(bytes) + " > " + std::to_string(input_mem_[i].size()) +
                        ")");
  }
  std::memcpy(input_mem_[i].data(), data, bytes);
  input_mem_[i].cleanCache();
}

void Engine::infer(int timeout_ms) {
  Task task;
  BCDL_CHECK(hbDNNInferV2(task.addr(), outputs_.data(), inputs_.data(), dnn_));
  task.submit();
  task.wait(timeout_ms);
  // Device wrote the outputs; drop stale CPU cache lines before anyone reads.
  for (const auto& m : output_mem_) m.invalidateCache();
}

std::vector<int> Engine::shapeOf(const hbDNNTensorProperties& p) {
  std::vector<int> dims;
  dims.reserve(p.validShape.numDimensions);
  for (int d = 0; d < p.validShape.numDimensions; ++d) {
    dims.push_back(p.validShape.dimensionSize[d]);
  }
  return dims;
}

std::size_t Engine::elemSize(int tensor_type) {
  switch (tensor_type) {
    case HB_DNN_TENSOR_TYPE_S4:
    case HB_DNN_TENSOR_TYPE_U4:
      return 1;  // sub-byte; caller must handle packing
    case HB_DNN_TENSOR_TYPE_S8:
    case HB_DNN_TENSOR_TYPE_U8:
    case HB_DNN_TENSOR_TYPE_BOOL8:
      return 1;
    case HB_DNN_TENSOR_TYPE_F16:
    case HB_DNN_TENSOR_TYPE_S16:
    case HB_DNN_TENSOR_TYPE_U16:
      return 2;
    case HB_DNN_TENSOR_TYPE_F32:
    case HB_DNN_TENSOR_TYPE_S32:
    case HB_DNN_TENSOR_TYPE_U32:
      return 4;
    case HB_DNN_TENSOR_TYPE_F64:
    case HB_DNN_TENSOR_TYPE_S64:
    case HB_DNN_TENSOR_TYPE_U64:
      return 8;
    default:
      return 0;
  }
}

}  // namespace bcdl
