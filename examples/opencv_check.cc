// 验证板上 libopencv(mamba 装于 bcdl 环境)能被 bcdl 的工具链链接调用。
// 调用 core/imgproc 的关键函数 —— 这些正是接下来要替换自写实现的:
//   cv::resize(双线性) · cv::minAreaRect(旋转外接矩形) ·
//   cv::rotatedRectangleIntersection(旋转框求交,给 OBB / OCR 用)。
//
//   ./opencv_check

#include <cstdio>
#include <vector>

#include <opencv2/opencv.hpp>
// OpenCV 5 拆出了独立的 geometry 模块:minAreaRect / convexHull /
// rotatedRectangleIntersection 等在此,opencv.hpp 不自动包含。
#include <opencv2/geometry/2d.hpp>

int main() {
  std::printf("OpenCV %s\n", CV_VERSION);

  // 双线性 resize(替换 instance_seg / OCR 的自写 bilinearResize)
  cv::Mat a = cv::Mat::ones(4, 4, CV_32F), b;
  cv::resize(a, b, cv::Size(8, 8), 0, 0, cv::INTER_LINEAR);
  std::printf("resize 4x4 -> %dx%d  ok\n", b.cols, b.rows);

  // 最小外接旋转矩形(给 OCR DBNet 旋转四点框)
  std::vector<cv::Point2f> pts = {{0, 0}, {10, 0}, {10, 6}, {0, 6}};
  cv::RotatedRect rr = cv::minAreaRect(pts);
  std::printf("minAreaRect size=%.0fx%.0f angle=%.1f  ok\n", rr.size.width,
              rr.size.height, rr.angle);

  // 旋转矩形求交(给 OBB 旋转 NMS / IoU)
  cv::RotatedRect r1({5, 5}, {6, 4}, 30.f), r2({6, 5}, {6, 4}, -10.f);
  std::vector<cv::Point2f> inter;
  int st = cv::rotatedRectangleIntersection(r1, r2, inter);
  std::printf("rotatedRectangleIntersection status=%d pts=%zu  ok\n", st, inter.size());

  std::printf("OK: bcdl 工具链可链接调用 OpenCV(core/imgproc)。\n");
  return 0;
}
