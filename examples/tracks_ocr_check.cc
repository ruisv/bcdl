// 合成数据验证:ByteTrack 跟踪 + OCR 的 CTC 解码 + DBNet 连通域后处理。
// 这三者都是纯算法(不依赖模型),用合成输入做确定性验证。
//
//   ./tracks_ocr_check
//
// 退出码 0 = 全部通过。

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "bcdl/preproc/geometry.h"
#include "bcdl/tasks/detection.h"
#include "bcdl/tasks/ocr.h"
#include "bcdl/tracks/byte_tracker.h"

namespace {

int g_fail = 0;
void check(bool ok, const char* msg) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg);
  if (!ok) ++g_fail;
}

bcdl::Detection box(float cx, float cy, float s = 25.f, float score = 0.9f, int cls = 0) {
  return {cx - s, cy - s, cx + s, cy + s, score, cls};
}

// ---- ByteTrack:3 个匀速移动目标,中间一帧漏检 B,验证 id 稳定/重关联 -------
void test_bytetrack() {
  std::printf("== ByteTrack(3 移动目标 + 漏检重关联)==\n");
  bcdl::ByteTracker tracker;
  // 真值轨迹:A 右移、B 下移、C 左移。
  auto truth = [](int t, int who, float& cx, float& cy) {
    if (who == 0) { cx = 100 + 12.f * t; cy = 120; }       // A
    else if (who == 1) { cx = 320; cy = 120 + 11.f * t; }  // B
    else { cx = 540 - 12.f * t; cy = 360; }                // C
  };

  int id_of[3] = {-1, -1, -1};   // 各目标最近邻关联到的 track_id(稳定段)
  bool id_stable[3] = {true, true, true};
  int last_count = 0;
  const int FRAMES = 16, DROP_FRAME = 8;

  for (int t = 0; t < FRAMES; ++t) {
    std::vector<bcdl::Detection> dets;
    for (int who = 0; who < 3; ++who) {
      if (t == DROP_FRAME && who == 1) continue;  // 第 8 帧漏检 B
      float cx, cy; truth(t, who, cx, cy);
      dets.push_back(box(cx, cy));
    }
    std::vector<bcdl::Track> tracks = tracker.update(dets);

    // 把输出 track 用最近邻关联回真值目标,记录 id。
    for (int who = 0; who < 3; ++who) {
      if (t == DROP_FRAME && who == 1) continue;
      float cx, cy; truth(t, who, cx, cy);
      int best = -1; float bestd = 1e9f;
      for (size_t k = 0; k < tracks.size(); ++k) {
        const float tcx = (tracks[k].x1 + tracks[k].x2) / 2;
        const float tcy = (tracks[k].y1 + tracks[k].y2) / 2;
        const float d = std::hypot(tcx - cx, tcy - cy);
        if (d < bestd) { bestd = d; best = static_cast<int>(k); }
      }
      if (best >= 0 && bestd < 40.f && t >= 2) {  // 跳过前 2 帧的激活期
        const int id = tracks[best].track_id;
        if (id_of[who] == -1) id_of[who] = id;
        else if (id_of[who] != id) id_stable[who] = false;
      }
    }
    last_count = static_cast<int>(tracks.size());
    std::printf("  frame %2d: dets=%zu tracks=%d\n", t, dets.size(), last_count);
  }

  check(id_of[0] != -1 && id_stable[0], "目标 A 全程同一 track_id");
  check(id_of[1] != -1 && id_stable[1], "目标 B 漏检前后同一 track_id(重关联)");
  check(id_of[2] != -1 && id_stable[2], "目标 C 全程同一 track_id");
  const bool distinct = id_of[0] != id_of[1] && id_of[1] != id_of[2] && id_of[0] != id_of[2];
  check(distinct, "三个目标的 track_id 互不相同");
  check(last_count == 3, "末帧稳定为 3 条轨迹");
}

// ---- OCR CTC 贪心解码:已知 argmax 序列 -> 已知字符串 ----------------------
void test_ctc() {
  std::printf("== OCR CTC 贪心解码 ==\n");
  // dict[0]=blank, 1=a, 2=b, 3=c, 4=d
  std::vector<std::string> dict = {"<blank>", "a", "b", "c", "d"};
  const int T = 6, C = 5;
  // 每 step 的 argmax: a, a(重复), blank, b, b(重复), c  => "abc"
  const int peaks[T] = {1, 1, 0, 2, 2, 3};
  std::vector<float> logits(static_cast<size_t>(T) * C, 0.f);
  for (int t = 0; t < T; ++t) logits[t * C + peaks[t]] = 5.0f;  // softmax 峰值

  bcdl::RecResult r = bcdl::decodeCtc(logits.data(), T, C, dict);
  std::printf("  decoded=\"%s\" score=%.3f\n", r.text.c_str(), r.score);
  check(r.text == "abc", "CTC 去重去 blank => \"abc\"");
  check(r.score > 0.9f, "score 为被选 step 的平均 max 概率(应≈1)");
}

// ---- OCR DBNet 连通域:合成概率图含 2 个矩形块 -> 2 个框 -------------------
void test_dbnet() {
  std::printf("== OCR DBNet 连通域 + unclip ==\n");
  const int H = 40, W = 60;
  std::vector<float> prob(static_cast<size_t>(H) * W, 0.1f);
  auto fill = [&](int r0, int r1, int c0, int c1) {
    for (int r = r0; r < r1; ++r)
      for (int c = c0; c < c1; ++c) prob[r * W + c] = 0.9f;
  };
  fill(5, 15, 10, 25);    // 块1:中心约 (17, 10)
  fill(20, 33, 35, 55);   // 块2:中心约 (45, 26)

  bcdl::DbConfig cfg;  // box_thresh 0.6, unclip 1.5, min_size 3
  bcdl::LetterboxInfo lb = bcdl::computeLetterbox(W, H, W, H);  // identity 映射
  std::vector<bcdl::TextBox> boxes = bcdl::decodeDbnet(prob.data(), H, W, cfg, lb);

  std::printf("  得到 %zu 个文本框:\n", boxes.size());
  for (const auto& b : boxes)
    std::printf("    [%.0f,%.0f,%.0f,%.0f] score=%.2f\n", b.x1, b.y1, b.x2, b.y2, b.score);
  check(boxes.size() == 2, "两个分离矩形块 => 2 个框");

  // 验证两个框分别覆盖两块中心。
  auto covers = [&](float cx, float cy) {
    for (const auto& b : boxes)
      if (cx >= b.x1 && cx <= b.x2 && cy >= b.y1 && cy <= b.y2) return true;
    return false;
  };
  check(covers(17, 10), "框覆盖块1中心 (17,10)");
  check(covers(45, 26), "框覆盖块2中心 (45,26)");
}

}  // namespace

int main() {
  test_bytetrack();
  test_ctc();
  test_dbnet();
  std::printf("\n%s\n", g_fail == 0 ? "ALL PASS" : "SOME CHECKS FAILED");
  return g_fail == 0 ? 0 : 1;
}
