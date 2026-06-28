// Minimal M0 example: load an .hbm, print its I/O signature, run one inference
// with a zero-filled input, and report latency.
//
//   ./det_infer model.hbm [model_name]

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bcdl/bcdl.h"

namespace {

const char* typeName(int t) {
  switch (t) {
    case HB_DNN_TENSOR_TYPE_S8: return "s8";
    case HB_DNN_TENSOR_TYPE_U8: return "u8";
    case HB_DNN_TENSOR_TYPE_F16: return "f16";
    case HB_DNN_TENSOR_TYPE_S16: return "s16";
    case HB_DNN_TENSOR_TYPE_U16: return "u16";
    case HB_DNN_TENSOR_TYPE_F32: return "f32";
    case HB_DNN_TENSOR_TYPE_S32: return "s32";
    case HB_DNN_TENSOR_TYPE_U32: return "u32";
    default: return "?";
  }
}

void printTensor(const char* tag, int i, const std::vector<int>& shape, int type,
                 long aligned) {
  std::printf("  %s[%d]  shape=[", tag, i);
  for (size_t d = 0; d < shape.size(); ++d) {
    std::printf("%d%s", shape[d], d + 1 < shape.size() ? "," : "");
  }
  std::printf("]  dtype=%s  aligned=%ld B\n", typeName(type), aligned);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s model.hbm [model_name]\n", argv[0]);
    return 1;
  }
  try {
    bcdl::Engine engine(argv[1], argc > 2 ? argv[2] : "");
    std::printf("model: %s\n", engine.modelName().c_str());

    for (int i = 0; i < engine.numInputs(); ++i) {
      printTensor("in ", i, engine.inputShape(i), engine.inputType(i),
                  static_cast<long>(engine.inputBytes(i)));
    }
    for (int i = 0; i < engine.numOutputs(); ++i) {
      printTensor("out", i, engine.outputShape(i), engine.outputType(i),
                  engine.outputProperties(i).alignedByteSize);
    }

    // Zero-fill all inputs.
    for (int i = 0; i < engine.numInputs(); ++i) {
      std::vector<unsigned char> zeros(engine.inputBytes(i), 0);
      engine.setInput(i, zeros.data(), zeros.size());
    }

    // Warm-up + timed runs.
    engine.infer();
    const int iters = 50;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < iters; ++n) engine.infer();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    std::printf("infer: %.3f ms/frame  (%.1f FPS) over %d iters\n", ms, 1000.0 / ms,
                iters);

    // Peek first floats of output 0 if it is F32.
    if (engine.numOutputs() > 0 &&
        engine.outputType(0) == HB_DNN_TENSOR_TYPE_F32) {
      const float* o = static_cast<const float*>(engine.outputData(0));
      std::printf("out[0][:8] =");
      for (int k = 0; k < 8; ++k) std::printf(" %.4f", o[k]);
      std::printf("\n");
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  return 0;
}
