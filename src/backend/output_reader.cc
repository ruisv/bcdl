#include "bcdl/backend/output_reader.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

#include "bcdl/backend/engine.h"
#include "bcdl/core/status.h"
#include "hobot/dnn/hb_dnn.h"

namespace bcdl {

namespace {

/// IEEE-754 binary16 -> binary32. hbDNN F16 outputs are little-endian uint16.
inline float half2float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +/- zero
    } else {
      // subnormal -> normalize
      int e = -1;
      uint32_t m = mant;
      do {
        m <<= 1;
        ++e;
      } while ((m & 0x400u) == 0);
      m &= 0x3FFu;
      const uint32_t fexp = static_cast<uint32_t>(127 - 15 - e);
      bits = sign | (fexp << 23) | (m << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
  } else {
    const uint32_t fexp = exp - 15 + 127;
    bits = sign | (fexp << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

/// Read element `linear` (a logical row-major flat index over validShape) from a
/// raw device buffer, honoring byte strides and dequantizing per the tensor's
/// quant params. `dims`/`strides` come straight from hbDNNTensorProperties.
float readElement(const uint8_t* base, int tensor_type, const hbDNNQuantiScale& q,
                  hbDNNQuantiType quanti, int quant_axis, int num_dims,
                  const int32_t* dims, const int64_t* strides, int64_t linear) {
  // Decompose the linear index into per-dim coordinates and accumulate the byte
  // offset; track the coordinate along the quant axis for per-channel scale.
  int64_t byte_off = 0;
  int64_t rem = linear;
  int axis_coord = 0;
  for (int d = num_dims - 1; d >= 0; --d) {
    const int64_t coord = dims[d] > 0 ? (rem % dims[d]) : 0;
    rem = dims[d] > 0 ? (rem / dims[d]) : rem;
    byte_off += coord * strides[d];
    if (d == quant_axis) axis_coord = static_cast<int>(coord);
  }
  const uint8_t* p = base + byte_off;

  // Read the raw numeric value at native precision.
  double raw;
  switch (tensor_type) {
    case HB_DNN_TENSOR_TYPE_F32: {
      float v;
      std::memcpy(&v, p, sizeof(v));
      return v;  // already float, never quantized
    }
    case HB_DNN_TENSOR_TYPE_F16: {
      uint16_t v;
      std::memcpy(&v, p, sizeof(v));
      return half2float(v);
    }
    case HB_DNN_TENSOR_TYPE_S8: {
      int8_t v;
      std::memcpy(&v, p, sizeof(v));
      raw = v;
      break;
    }
    case HB_DNN_TENSOR_TYPE_U8: {
      uint8_t v;
      std::memcpy(&v, p, sizeof(v));
      raw = v;
      break;
    }
    case HB_DNN_TENSOR_TYPE_S16: {
      int16_t v;
      std::memcpy(&v, p, sizeof(v));
      raw = v;
      break;
    }
    case HB_DNN_TENSOR_TYPE_U16: {
      uint16_t v;
      std::memcpy(&v, p, sizeof(v));
      raw = v;
      break;
    }
    case HB_DNN_TENSOR_TYPE_S32: {
      int32_t v;
      std::memcpy(&v, p, sizeof(v));
      raw = v;
      break;
    }
    case HB_DNN_TENSOR_TYPE_U32: {
      uint32_t v;
      std::memcpy(&v, p, sizeof(v));
      raw = v;
      break;
    }
    default:
      throw Error(-1, "BCDL output_reader: unsupported output tensor type " +
                          std::to_string(tensor_type) +
                          "; export an F32 (or F16) head");
  }

  if (quanti != SCALE || q.scaleLen <= 0 || q.scaleData == nullptr) {
    // Integer output without dequant info — values are taken as-is.
    return static_cast<float>(raw);
  }
  // Per-tensor (scaleLen==1) or per-channel (indexed by the quant axis) scale.
  // Clamp the array index into [0, len): the per-channel scale/zero-point arrays
  // are sized by the quant axis, but nothing guarantees scaleLen equals the axis
  // dimension — a stale/mismatched quantizeAxis or shorter array would otherwise
  // read past the allocation (MEDIUM: out-of-bounds read on quantized output).
  const int si = q.scaleLen == 1 ? 0 : std::max(0, std::min(axis_coord, q.scaleLen - 1));
  double scaled = raw;
  if (q.zeroPointLen > 0 && q.zeroPointData != nullptr) {
    const int zi =
        q.zeroPointLen == 1 ? 0 : std::max(0, std::min(axis_coord, q.zeroPointLen - 1));
    scaled -= q.zeroPointData[zi];
  }
  scaled *= q.scaleData[si];
  return static_cast<float>(scaled);
}

/// True when a tensor is laid out fully packed (contiguous row-major, no stride
/// padding on any axis) — so its device buffer can be read straight as a float
/// array without an index-by-index stride walk.
bool isPackedRowMajor(const hbDNNTensorProperties& p) {
  const int n = p.validShape.numDimensions;
  int64_t packed = static_cast<int64_t>(Engine::elemSize(p.tensorType));
  for (int i = n - 1; i >= 0; --i) {
    if (p.stride[i] != packed) return false;
    packed *= p.validShape.dimensionSize[i];
  }
  return true;
}

}  // namespace

const float* outputAsFloat(Engine& engine, int out_idx, std::vector<float>& scratch,
                           std::vector<int>& shape) {
  shape = engine.outputShape(out_idx);
  const hbDNNTensorProperties& props = engine.outputProperties(out_idx);
  const auto* base = static_cast<const uint8_t*>(engine.outputData(out_idx));

  if (props.tensorType == HB_DNN_TENSOR_TYPE_F32 && isPackedRowMajor(props)) {
    return reinterpret_cast<const float*>(base);  // zero-copy
  }

  int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? d : 1);
  scratch.resize(static_cast<size_t>(total));
  const int num_dims = props.validShape.numDimensions;
  for (int64_t i = 0; i < total; ++i) {
    scratch[static_cast<size_t>(i)] =
        readElement(base, props.tensorType, props.scale, props.quantiType,
                    props.quantizeAxis, num_dims, props.validShape.dimensionSize,
                    props.stride, i);
  }
  return scratch.data();
}

}  // namespace bcdl
