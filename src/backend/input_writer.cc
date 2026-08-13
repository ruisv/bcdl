#include "bcdl/backend/input_writer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include "bcdl/backend/engine.h"
#include "bcdl/core/status.h"
#include "hobot/dnn/hb_dnn.h"

namespace bcdl {

namespace {

/// Quantized store of one element: the exact inverse of output_reader's
/// readElement() — subtract nothing, scale down, round, add the zero point,
/// clamp. Rounding is half-away-from-zero (std::lround), matching numpy's
/// np.round on the .5 cases that matter here only at the LSB.
template <typename T>
void storeQuantized(std::uint8_t* dst, std::size_t index, double value, double scale,
                    double zero_point) {
  double q = (scale > 0.0 ? value / scale : value) + zero_point;
  q = std::round(q);
  const double lo = static_cast<double>(std::numeric_limits<T>::min());
  const double hi = static_cast<double>(std::numeric_limits<T>::max());
  const T out = static_cast<T>(std::min(hi, std::max(lo, q)));
  std::memcpy(dst + index * sizeof(T), &out, sizeof(T));
}

/// Coordinate along the quantization axis for a logical row-major flat index —
/// only needed when the tensor carries a per-channel scale.
int axisCoord(int num_dims, const int32_t* dims, int quant_axis, std::size_t linear) {
  if (quant_axis < 0 || quant_axis >= num_dims) return 0;
  std::size_t rem = linear;
  int coord = 0;
  for (int d = num_dims - 1; d >= 0; --d) {
    const int size = dims[d] > 0 ? dims[d] : 1;
    const int c = static_cast<int>(rem % static_cast<std::size_t>(size));
    rem /= static_cast<std::size_t>(size);
    if (d == quant_axis) coord = c;
  }
  return coord;
}

}  // namespace

std::size_t inputElementCount(const Engine& engine, int in_idx) {
  std::size_t total = 1;
  for (int d : engine.inputShape(in_idx)) {
    total *= static_cast<std::size_t>(d > 0 ? d : 1);
  }
  return total;
}

void setInputFromFloat(Engine& engine, int in_idx, const float* data, std::size_t count,
                       std::vector<std::uint8_t>& scratch) {
  const hbDNNTensorProperties& props = engine.inputProperties(in_idx);
  const std::size_t expected = inputElementCount(engine, in_idx);
  if (count != expected) {
    throw Error(-1, "BCDL setInputFromFloat: input " + std::to_string(in_idx) +
                        " wants " + std::to_string(expected) + " elements, got " +
                        std::to_string(count));
  }

  const int type = props.tensorType;
  if (type == HB_DNN_TENSOR_TYPE_F32) {
    engine.setInput(in_idx, data, count * sizeof(float));
    return;
  }
  if (type == HB_DNN_TENSOR_TYPE_F16) {
    throw Error(-1,
                "BCDL setInputFromFloat: input " + std::to_string(in_idx) +
                    " is F16; no float->half conversion here. Write the half bytes "
                    "yourself and call Engine::setInput().");
  }

  const std::size_t elem = Engine::elemSize(type);
  scratch.resize(count * elem);
  std::uint8_t* dst = scratch.data();

  const hbDNNQuantiScale& q = props.scale;
  const bool has_scale =
      props.quantiType == SCALE && q.scaleLen > 0 && q.scaleData != nullptr;
  const bool per_axis = has_scale && q.scaleLen > 1;
  const int num_dims = props.validShape.numDimensions;

  for (std::size_t i = 0; i < count; ++i) {
    double scale = 1.0, zp = 0.0;
    if (has_scale) {
      const int coord =
          per_axis ? axisCoord(num_dims, props.validShape.dimensionSize, props.quantizeAxis, i)
                   : 0;
      const int si = q.scaleLen == 1 ? 0 : std::max(0, std::min(coord, q.scaleLen - 1));
      scale = q.scaleData[si];
      if (q.zeroPointLen > 0 && q.zeroPointData != nullptr) {
        const int zi =
            q.zeroPointLen == 1 ? 0 : std::max(0, std::min(coord, q.zeroPointLen - 1));
        zp = q.zeroPointData[zi];
      }
    }
    const double v = data[i];
    switch (type) {
      case HB_DNN_TENSOR_TYPE_S8:
        storeQuantized<std::int8_t>(dst, i, v, scale, zp);
        break;
      case HB_DNN_TENSOR_TYPE_U8:
        storeQuantized<std::uint8_t>(dst, i, v, scale, zp);
        break;
      case HB_DNN_TENSOR_TYPE_S16:
        storeQuantized<std::int16_t>(dst, i, v, scale, zp);
        break;
      case HB_DNN_TENSOR_TYPE_U16:
        storeQuantized<std::uint16_t>(dst, i, v, scale, zp);
        break;
      case HB_DNN_TENSOR_TYPE_S32:
        storeQuantized<std::int32_t>(dst, i, v, scale, zp);
        break;
      case HB_DNN_TENSOR_TYPE_U32:
        storeQuantized<std::uint32_t>(dst, i, v, scale, zp);
        break;
      default:
        throw Error(-1, "BCDL setInputFromFloat: unsupported input tensor type " +
                            std::to_string(type));
    }
  }

  engine.setInput(in_idx, scratch.data(), scratch.size());
}

}  // namespace bcdl
