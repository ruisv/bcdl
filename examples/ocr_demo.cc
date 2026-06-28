// OCR end-to-end demo on the LATEST PP-OCRv5 .hbm models (converted offline from
// ccdl ONNX; see docs/PLAN.md finding #6). Three-stage PaddleOCR pipeline routed
// through bcdl's task layer, so this runs the real models AND validates
// bcdl::DbTextDetector / TextAngleClassifier / TextRecognizer end-to-end:
//
//   1. DET — PP-OCRv5_server_det (DBNet). Input: F32 RGB NCHW [1,3,960,960]
//      (ImageNet norm). Output: sigmoid prob map [1,1,960,960] -> bcdl::DbTextDetector
//      -> rotated 4-point text boxes.
//   2. CLS — PP-LCNet textline ori (OPTIONAL). Input [1,3,80,160] -> [1,2]; flips
//      the crop 180° when it reads upside-down.
//   3. REC — PP-OCRv5_server_rec (CRNN+CTC). Input F32 RGB NCHW [1,3,48,320]
//      (0.5/0.5 norm). Output [1,40,18385] softmax -> CTC greedy text (18385-class
//      v5 dict).
//
// Preprocessing follows ccdl/PaddleOCR source (get_rotate_crop_image /
// order_points_clockwise / resize_norm_img):
//   det : anamorphic resize to 960x960, RGB, ImageNet z-score, NCHW F32.
//   crop: reading-order corners (order_points_clockwise) + perspective rectify;
//         tall lines rotate 90° CCW (np.rot90).
//   rec/cls : aspect-ratio-preserving resize to the model height + right-pad to
//             the model width (not an anamorphic stretch).
// Box geometry: decoded in det-pixel space (identity letterbox) and scaled back to
// the original image anamorphically (sx=W/960, sy=H/960). The box *extraction*
// deviates from PaddleOCR on purpose (bcdl decodeDbnet: binarise at
// DbConfig.bin_thresh + min-area-rect unclip), so the thresholds below are tuned.
//
//   ./ocr_demo <image.jpg> [det.hbm] [rec.hbm] [dict.txt] [out.jpg] [cls.hbm]
//
// Defaults are repo-relative (models/ + data/), so from the repo root on the board
// `./build/ocr_demo data/images/ocr.jpg` just works.

#include <algorithm>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/geometry/2d.hpp>  // OpenCV 5: minAreaRect lives here
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>      // getPerspectiveTransform, warpPerspective, resize

#include "bcdl/backend/engine.h"
#include "bcdl/tasks/ocr.h"

namespace {

// LATEST PP-OCRv5 stack (converted offline from ccdl ONNX). All single-input F32
// RGB NCHW featuremap models; normalisation done in preproc here (det/cls
// ImageNet, rec 0.5/0.5). Paths are repo-relative (run from the repo root, like
// the dict) — models/ is populated by scripts/fetch_models.sh.
constexpr const char* kDefDet = "models/ppocrv5_server_det_960x960.hbm";
constexpr const char* kDefRec = "models/ppocrv5_server_rec_48x320.hbm";
constexpr const char* kDefDict = "data/ppocr_keys_v5_18385.txt";
// PP-LCNet textline direction classifier (80x160). OPTIONAL: the demo skips the
// 180° flip stage when this model is absent.
constexpr const char* kDefCls = "models/ppocrv5_lcnet_cls_80x160.hbm";

// PaddleOCR get_rotate_crop_image: order the 4 box corners as TL,TR,BR,BL,
// perspective-rectify to an upright (w x h) crop, then rotate 90 CW for tall
// (vertical) text. Ordering the corners ourselves makes this independent of the
// minAreaRect angle/size convention (which differs between OpenCV 4 and 5 — that
// mismatch transposed the crops and produced all-blank recognition).
cv::Mat cropAndRotate(const cv::Mat& img, const float pts[8]) {
  cv::Point2f p[4] = {{pts[0], pts[1]}, {pts[2], pts[3]},
                      {pts[4], pts[5]}, {pts[6], pts[7]}};
  // order_points_clockwise (ccdl / PaddleOCR get_mini_boxes): sort by x, split
  // each side by y -> [LT, RT, RB, LB]. Robust for slanted boxes (the x+y / y-x
  // heuristic mis-assigns corners once the box tilts past ~45°).
  std::sort(p, p + 4, [](cv::Point2f a, cv::Point2f b) { return a.x < b.x; });
  cv::Point2f lt = p[0], lb = p[1], rt = p[2], rb = p[3];
  if (lt.y > lb.y) std::swap(lt, lb);
  if (rt.y > rb.y) std::swap(rt, rb);
  const cv::Point2f src[4] = {lt, rt, rb, lb};
  const auto dist = [](cv::Point2f a, cv::Point2f b) { return std::hypot(a.x - b.x, a.y - b.y); };
  const int w = static_cast<int>(std::max(dist(src[0], src[1]), dist(src[3], src[2])));
  const int h = static_cast<int>(std::max(dist(src[0], src[3]), dist(src[1], src[2])));
  if (w <= 0 || h <= 0) return cv::Mat();

  const cv::Point2f dst[4] = {{0.0f, 0.0f},
                              {static_cast<float>(w), 0.0f},
                              {static_cast<float>(w), static_cast<float>(h)},
                              {0.0f, static_cast<float>(h)}};
  const cv::Mat M = cv::getPerspectiveTransform(src, dst);
  cv::Mat warped;
  cv::warpPerspective(img, warped, M, cv::Size(w, h), cv::INTER_CUBIC,
                      cv::BORDER_REPLICATE);
  // PaddleOCR get_rotate_crop_image uses np.rot90 (counter-clockwise); rotating
  // clockwise leaves vertical lines upside-down (the weak direction-cls can't
  // always recover them, e.g. short Latin "ODM OEM").
  if (static_cast<float>(h) / static_cast<float>(w) >= 1.5f)
    cv::rotate(warped, warped, cv::ROTATE_90_COUNTERCLOCKWISE);
  return warped;
}

// Resize to (W,H), BGR->RGB, normalise, pack into F32 NCHW [1,3,H,W]. `mean`/`inv`
// are per-channel: out = (x/255 - mean) * inv. Shared by det/rec/cls.
void packNchw(const cv::Mat& crop, int H, int W, const float mean[3],
              const float inv[3], std::vector<float>& buf) {
  cv::Mat resized;
  cv::resize(crop, resized, cv::Size(W, H));  // anamorphic, matches reference
  buf.assign(static_cast<size_t>(3) * H * W, 0.0f);
  const int plane = H * W;
  for (int y = 0; y < H; ++y) {
    const auto* row = resized.ptr<uint8_t>(y);
    for (int x = 0; x < W; ++x) {
      const uint8_t b = row[x * 3 + 0], g = row[x * 3 + 1], r = row[x * 3 + 2];
      const int idx = y * W + x;
      buf[0 * plane + idx] = (r / 255.0f - mean[0]) * inv[0];  // R
      buf[1 * plane + idx] = (g / 255.0f - mean[1]) * inv[1];  // G
      buf[2 * plane + idx] = (b / 255.0f - mean[2]) * inv[2];  // B
    }
  }
}

// ImageNet z-score (det + PP-LCNet cls); rec uses 0.5/0.5 -> [-1,1].
constexpr float kImnMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImnInv[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};
constexpr float kHalf[3] = {0.5f, 0.5f, 0.5f};
constexpr float kTwo[3] = {2.0f, 2.0f, 2.0f};  // 1/0.5

// Aspect-ratio-preserving resize to height H (width capped at W) + right-pad to W
// with 0 in normalised space (PaddleOCR resize_norm_img / ccdl resize_img_with_pad).
// Avoids the character distortion of an anamorphic stretch on rec/cls crops.
void packNchwPad(const cv::Mat& crop, int H, int W, const float mean[3],
                 const float inv[3], std::vector<float>& buf) {
  const float ratio = static_cast<float>(crop.cols) / std::max(crop.rows, 1);
  int rw = (static_cast<int>(std::ceil(H * ratio)) > W)
               ? W
               : static_cast<int>(std::ceil(H * ratio));
  rw = std::max(1, std::min(rw, W));
  cv::Mat resized;
  cv::resize(crop, resized, cv::Size(rw, H));
  buf.assign(static_cast<size_t>(3) * H * W, 0.0f);  // pad = 0 (normalised space)
  const int plane = H * W;
  for (int y = 0; y < H; ++y) {
    const auto* row = resized.ptr<uint8_t>(y);
    for (int x = 0; x < rw; ++x) {
      const uint8_t b = row[x * 3 + 0], g = row[x * 3 + 1], r = row[x * 3 + 2];
      const int idx = y * W + x;
      buf[0 * plane + idx] = (r / 255.0f - mean[0]) * inv[0];
      buf[1 * plane + idx] = (g / 255.0f - mean[1]) * inv[1];
      buf[2 * plane + idx] = (b / 255.0f - mean[2]) * inv[2];
    }
  }
}

void detPreprocess(const cv::Mat& bgr, int H, int W, std::vector<float>& buf) {
  packNchw(bgr, H, W, kImnMean, kImnInv, buf);            // det: anamorphic to HxW
}
void recPreprocess(const cv::Mat& crop, int RH, int RW, std::vector<float>& buf) {
  packNchwPad(crop, RH, RW, kHalf, kTwo, buf);
}
void clsPreprocess(const cv::Mat& crop, int CH, int CW, std::vector<float>& buf) {
  packNchwPad(crop, CH, CW, kImnMean, kImnInv, buf);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <image.jpg> [det.hbm] [rec.hbm] [dict.txt] [out.jpg] [cls.hbm]\n",
                 argv[0]);
    return 1;
  }
  const std::string image_path = argv[1];
  const std::string det_path = argc > 2 ? argv[2] : kDefDet;
  const std::string rec_path = argc > 3 ? argv[3] : kDefRec;
  const std::string dict_path = argc > 4 ? argv[4] : kDefDict;
  const std::string out_path = argc > 5 ? argv[5] : "ocr_result.jpg";
  const std::string cls_path = argc > 6 ? argv[6] : kDefCls;

  try {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);  // BGR
    if (img.empty()) throw std::runtime_error("cannot read image: " + image_path);
    const int img_w = img.cols, img_h = img.rows;

    // ----- DET: PP-OCRv5 server, F32 RGB NCHW input [1,3,H,W] (960x960).
    bcdl::Engine det(det_path);
    const std::vector<int> ds = det.inputShape(0);  // NCHW: [1,3,H,W]
    const int DH = ds.size() == 4 ? ds[2] : 960;
    const int DW = ds.size() == 4 ? ds[3] : 960;
    std::printf("det: %s  in=%dx%d  outputs=%d\n", det.modelName().c_str(), DW, DH,
                det.numOutputs());

    // Anamorphic resize -> DWxDH RGB, ImageNet z-score, NCHW F32 (reference pre).
    std::vector<float> detbuf;
    detPreprocess(img, DH, DW, detbuf);
    det.setInput(0, detbuf.data(), detbuf.size() * sizeof(float));
    det.infer();

    // Decode in det-pixel space (identity letterbox), then scale boxes to the
    // original image anamorphically. The DB sigmoid map is already [0,1]; PP-OCRv5
    // server defaults: bin 0.3 / box 0.6 / unclip 1.5. Override via OCR_BIN/BOX/UNCLIP.
    bcdl::DbConfig dbcfg;
    dbcfg.bin_thresh = 0.3f;
    dbcfg.box_thresh = 0.6f;
    dbcfg.unclip_ratio = 1.5f;
    dbcfg.min_size = 3;
    if (const char* e = std::getenv("OCR_BIN")) dbcfg.bin_thresh = std::atof(e);
    if (const char* e = std::getenv("OCR_BOX")) dbcfg.box_thresh = std::atof(e);
    if (const char* e = std::getenv("OCR_UNCLIP")) dbcfg.unclip_ratio = std::atof(e);
    bcdl::DbTextDetector detector(det, dbcfg, /*out_idx=*/0);
    std::vector<bcdl::TextBox> boxes = detector.postprocess(bcdl::LetterboxInfo{});

    const float sx = static_cast<float>(img_w) / DW;
    const float sy = static_cast<float>(img_h) / DH;
    for (auto& b : boxes) {
      for (int k = 0; k < 4; ++k) {
        b.pts[2 * k] = std::min(std::max(b.pts[2 * k] * sx, 0.0f),
                                static_cast<float>(img_w - 1));
        b.pts[2 * k + 1] = std::min(std::max(b.pts[2 * k + 1] * sy, 0.0f),
                                    static_cast<float>(img_h - 1));
      }
    }
    std::printf("det: %zu text boxes\n", boxes.size());

    // ----- REC: one CRNN+CTC pass per cropped/rectified text line.
    bcdl::Engine rec(rec_path);
    const std::vector<int> rs = rec.inputShape(0);  // NCHW [1,3,H,W]
    const int RH = rs.size() == 4 ? rs[2] : 48;
    const int RW = rs.size() == 4 ? rs[3] : 320;
    const size_t rec_bytes = static_cast<size_t>(3) * RH * RW * sizeof(float);
    if (rec.inputBytes(0) != rec_bytes) {
      std::fprintf(stderr, "warn: rec input bytes %zu != expected %zu\n",
                   rec.inputBytes(0), rec_bytes);
    }
    bcdl::TextRecognizer recognizer(rec, dict_path, /*out_idx=*/0);
    std::printf("rec: %s  in=%dx%d  dict=%zu classes\n", rec.modelName().c_str(), RW,
                RH, recognizer.dict().size());

    // ----- CLS (optional): 0°/180° text-line direction. When deployed, each
    // crop is checked and flipped before rec so upside-down lines recognise.
    std::unique_ptr<bcdl::Engine> cls;
    std::unique_ptr<bcdl::TextAngleClassifier> angler;
    int CH = 48, CW = 192;
    if (std::ifstream(cls_path).good()) {
      cls = std::make_unique<bcdl::Engine>(cls_path);
      const std::vector<int> cs = cls->inputShape(0);  // NCHW [1,3,H,W]
      CH = cs.size() == 4 ? cs[2] : 48;
      CW = cs.size() == 4 ? cs[3] : 192;
      angler = std::make_unique<bcdl::TextAngleClassifier>(*cls, /*thresh=*/0.9f, 0);
      std::printf("cls: %s  in=%dx%d (0/180 direction)\n", cls->modelName().c_str(),
                  CW, CH);
    } else {
      std::printf("cls: (none — det->rec only; deploy %s to enable)\n", cls_path.c_str());
    }

    const bool dbg = std::getenv("OCR_DEBUG") != nullptr;
    cv::Mat vis = img.clone();
    std::vector<float> buf;
    for (size_t i = 0; i < boxes.size(); ++i) {
      cv::Mat crop = cropAndRotate(img, boxes[i].pts);
      if (dbg) {
        std::printf("  box[%zu] pts=(%.0f,%.0f)(%.0f,%.0f)(%.0f,%.0f)(%.0f,%.0f) crop=%dx%d\n",
                    i, boxes[i].pts[0], boxes[i].pts[1], boxes[i].pts[2], boxes[i].pts[3],
                    boxes[i].pts[4], boxes[i].pts[5], boxes[i].pts[6], boxes[i].pts[7],
                    crop.cols, crop.rows);
        if (!crop.empty()) cv::imwrite("crop_" + std::to_string(i) + ".jpg", crop);
      }
      if (crop.empty()) continue;
      if (angler) {                       // 180° flip stage (upside-down -> upright)
        clsPreprocess(crop, CH, CW, buf);
        cls->setInput(0, buf.data(), buf.size() * sizeof(float));
        cls->infer();
        if (angler->postprocess().flip180) cv::rotate(crop, crop, cv::ROTATE_180);
      }
      recPreprocess(crop, RH, RW, buf);
      rec.setInput(0, buf.data(), buf.size() * sizeof(float));
      rec.infer();
      if (dbg) {
        // Debug-only stats; reads the raw output buffer assuming a contiguous
        // [T,C] layout. The real decode path (TextRecognizer -> outputAsFloat)
        // handles any output stride — this block does not, so treat it as
        // indicative only.
        const float* lo = static_cast<const float*>(rec.outputData(0));
        const std::vector<int> osh = rec.outputShape(0);  // [1,T,C]
        const int T = osh.size() == 3 ? osh[1] : 40, C = osh.size() == 3 ? osh[2] : 6625;
        int nonblank = 0;
        float lmin = lo[0], lmax = lo[0];
        for (int t = 0; t < T; ++t) {
          int am = 0; float mv = lo[t * C];
          for (int c = 1; c < C; ++c) { float v = lo[t * C + c]; if (v > mv) { mv = v; am = c; } }
          if (am != 0) ++nonblank;
        }
        for (int k = 0; k < T * C; ++k) { lmin = std::min(lmin, lo[k]); lmax = std::max(lmax, lo[k]); }
        std::printf("    rec out [%d,%d] logit range [%.3f,%.3f] nonblank_steps=%d\n",
                    T, C, lmin, lmax, nonblank);
      }
      const bcdl::RecResult r = recognizer.postprocess();
      std::printf("[%zu] score=%.3f  text=%s\n", i, r.score, r.text.c_str());

      std::vector<cv::Point> poly(4);
      for (int k = 0; k < 4; ++k)
        poly[k] = {static_cast<int>(boxes[i].pts[2 * k]),
                   static_cast<int>(boxes[i].pts[2 * k + 1])};
      cv::polylines(vis, poly, true, cv::Scalar(0, 0, 255), 2);
    }

    cv::imwrite(out_path, vis);
    std::printf("OK: wrote %s (boxes drawn; recognised text on stdout)\n",
                out_path.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ocr_demo error: %s\n", e.what());
    return 2;
  }
  return 0;
}
