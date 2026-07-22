#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "bcdl/core/sys_mem.h"
#include "hobot/dnn/hb_dnn.h"

namespace bcdl {

/// Loads a compiled `.hbm` model and runs BPU inference through hbDNN + hbUCP.
///
/// Tensor buffers (one SysMem per input/output) are allocated once at
/// construction from the model's tensor properties, then reused every infer()
/// — no per-frame allocation. Cache coherency is handled internally:
/// setInput() cleans, infer() invalidates outputs before they are read.
class Engine {
 public:
  /// Load `hbm_path`. If the package holds multiple models, pick `model_name`
  /// (empty => first model).
  explicit Engine(const std::string& hbm_path, const std::string& model_name = "");
  ~Engine();

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  const std::string& modelName() const noexcept { return model_name_; }

  /// Every model name packed into `hbm_path`, without building an Engine.
  ///
  /// A `.hbm` is a PACKAGE and may hold several models — the official SigLIP
  /// encoders, for instance, expose a global-embedding submodel and a patch-
  /// feature submodel from one file. The constructor's `model_name` selects one
  /// (empty = the first), but there was no way to find out what to pass short of
  /// running `hrt_model_exec model_info` by hand. This is that.
  static std::vector<std::string> modelNames(const std::string& hbm_path);

  int numInputs() const noexcept { return static_cast<int>(inputs_.size()); }
  int numOutputs() const noexcept { return static_cast<int>(outputs_.size()); }

  const hbDNNTensorProperties& inputProperties(int i) const { return inputs_[i].properties; }
  const hbDNNTensorProperties& outputProperties(int i) const { return outputs_[i].properties; }

  std::vector<int> inputShape(int i) const { return shapeOf(inputs_[i].properties); }
  std::vector<int> outputShape(int i) const { return shapeOf(outputs_[i].properties); }
  int inputType(int i) const { return inputs_[i].properties.tensorType; }
  int outputType(int i) const { return outputs_[i].properties.tensorType; }

  /// Allocated byte size of input[i]'s device buffer (post stride-resolution).
  std::size_t inputBytes(int i) const { return input_mem_[i].size(); }

  /// Byte size of input[i] as a PACKED (contiguous, row-major) host array —
  /// product of the valid dims times the element size. Differs from
  /// inputBytes(i) whenever the model pads a dimension; see setInput().
  std::size_t inputPackedBytes(int i) const;

  /// Resolved byte strides of input[i], innermost last (dynamic -1 strides are
  /// resolved at load time). Anyone writing an input buffer directly instead of
  /// through setInput() must honor these.
  std::vector<std::int64_t> inputStride(int i) const;

  /// Resolved byte strides of output[i], innermost last. Outputs are padded too
  /// — a [1,133,64,48] F32 heatmap comes back with a 256-byte row stride, so a
  /// flat reshape of outputData() shears every map. outputAsFloat() handles this;
  /// anyone reading the buffer directly needs these.
  std::vector<std::int64_t> outputStride(int i) const;

  /// Copy `bytes` of host data into input[i]'s device buffer and flush.
  ///
  /// `bytes` must be either inputPackedBytes(i) — a contiguous row-major array,
  /// which is scattered row by row into the model's (possibly padded) layout —
  /// or inputBytes(i), data already laid out in the device stride. Anything else
  /// throws rather than being copied in as a short prefix: a model whose row
  /// stride is padded (ArcFace R50: 112 f32 = 448 B/row padded to 512) would
  /// silently take a packed array as garbage, shifting every row after the first.
  void setInput(int i, const void* data, std::size_t bytes);

  /// Read-only view of input[i]'s device buffer, in the device layout. For
  /// diagnostics — the way to see how setInput() laid a packed array out.
  const void* inputData(int i) const { return input_mem_[i].data(); }

  /// Submit + wait + invalidate output caches. timeout_ms == 0 blocks forever.
  void infer(int timeout_ms = 0);

  /// Output buffer after infer() (cache already invalidated). Note: this is the
  /// aligned device buffer; honor outputProperties(i).stride for non-contiguous
  /// layouts.
  const void* outputData(int i) const { return output_mem_[i].data(); }
  std::size_t outputBytes(int i) const { return output_mem_[i].size(); }

  /// Element size in bytes for an hbDNN tensor type (HB_DNN_TENSOR_TYPE_*).
  static std::size_t elemSize(int tensor_type);

 private:
  static std::vector<int> shapeOf(const hbDNNTensorProperties& p);
  void allocTensors();

  hbDNNPackedHandle_t packed_ = nullptr;
  hbDNNHandle_t dnn_ = nullptr;
  std::string model_name_;

  std::vector<hbDNNTensor> inputs_;
  std::vector<hbDNNTensor> outputs_;
  std::vector<SysMem> input_mem_;
  std::vector<SysMem> output_mem_;
};

}  // namespace bcdl
