"""One calibrated KITTI frame: the sweep, the camera image, and the truth.

The published PointPillars sample is a bare point cloud, so the only check it
can support is "does the box sit on some points". A calibrated frame supports
the two checks that actually settle whether a detector works:

  * **project the 3-D boxes into the camera image.** A box either wraps the car
    or it does not, and a person can see which in one glance. A top-down
    footprint cannot show that the height or the heading is wrong.
  * **compare against the labels.** With ground truth the figure stops saying
    "plausible" and starts saying matched, missed, or invented.

The dataset is not redistributed with this library — see docs/MODELS.md for how
to stage a frame. Everything here works from the four files KITTI ships per
frame and nothing else.
"""
import os

import numpy as np

CLASSES = ["Car", "Pedestrian", "Cyclist"]

# The 12 edges of a cuboid over the corner order used below, and the 4 corners
# of its front face — drawn separately because a box without a marked front is
# only right modulo pi.
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
FRONT = [0, 1, 5, 4]


class KittiFrame:
    """The four files KITTI ships for one frame, and the geometry between them.

    Coordinate systems, because every bug in this file would come from mixing
    them up:

      * **lidar** — x forward, y left, z up. The detector's boxes live here.
      * **camera** — x right, y down, z forward. The labels live here, and a
        label's y is the BOTTOM of its box, not the centre.
      * **image** — pixels, after `P2`.

    `R0_rect` is the rectifying rotation and `Tr_velo_to_cam` the extrinsic, so
    a lidar point reaches the image as `P2 @ R0 @ Tr @ p`.
    """

    def __init__(self, directory, name="frame"):
        self.dir, self.name = directory, name
        self.points = np.fromfile(self._f("velodyne.bin"), np.float32).reshape(-1, 4)
        self.P2, self.R0, self.Tr = self._calib(self._f("calib.txt"))
        self._image_path = self._f("image_2.png")
        self._label_path = os.path.join(directory, "label_2.txt")
        self._size = None
        fid = os.path.join(directory, "frame_id.txt")
        if os.path.exists(fid):
            self.name = open(fid).read().strip() or name

    # -- loading ------------------------------------------------------------
    def _f(self, base):
        p = os.path.join(self.dir, base)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p

    @staticmethod
    def _calib(path):
        raw = {}
        for line in open(path):
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            raw[k.strip()] = np.array([float(t) for t in v.split()])
        R0 = np.eye(4)
        R0[:3, :3] = raw["R0_rect"].reshape(3, 3)
        Tr = np.eye(4)
        Tr[:3, :4] = raw["Tr_velo_to_cam"].reshape(3, 4)
        return raw["P2"].reshape(3, 4), R0, Tr

    @staticmethod
    def available(directory):
        return directory and all(
            os.path.exists(os.path.join(directory, f))
            for f in ("velodyne.bin", "calib.txt", "image_2.png"))

    def image(self):
        import cv2

        img = cv2.imread(self._image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("cannot read %s" % self._image_path)
        return img

    # -- geometry -----------------------------------------------------------
    def project(self, xyz_lidar):
        """Lidar points -> (uv, depth). Depth is the CAMERA z, which is what
        decides whether a point is in front of the camera at all."""
        xyz = np.asarray(xyz_lidar, np.float64).reshape(-1, 3)
        h = np.hstack([xyz, np.ones((len(xyz), 1))])
        cam = (h @ self.Tr.T @ self.R0.T)[:, :3]
        p = np.hstack([cam, np.ones((len(cam), 1))]) @ self.P2.T
        return p[:, :2] / np.where(np.abs(p[:, 2:3]) < 1e-6, 1e-6, p[:, 2:3]), cam[:, 2]

    @staticmethod
    def box_corners(b):
        """A box -> its 8 corners in the LIDAR frame.

        `z` is the box CENTRE here, not its base — the same convention the
        anchor decode uses, and getting it wrong sinks every box half a car into
        the road without changing anything else about the picture.
        """
        c, s = np.cos(b.heading), np.sin(b.heading)
        d = np.array([[+1, +1, +1], [+1, -1, +1], [-1, -1, +1], [-1, +1, +1],
                      [+1, +1, -1], [+1, -1, -1], [-1, -1, -1], [-1, +1, -1]], np.float64)
        d = d * np.array([b.dx / 2.0, b.dy / 2.0, b.dz / 2.0])
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return d @ R.T + np.array([b.x, b.y, b.z])

    def in_camera(self, box, margin=0.0):
        """Is this box inside the camera's view?

        It has to be asked, because KITTI only labels what the camera can see
        while the detector is handed the whole 360-degree sweep. Score a
        prediction behind the car against labels that were never written for it
        and the figure reports false positives that are nothing of the sort —
        the benchmark's own evaluation restricts to this frustum for exactly
        this reason.
        """
        uv, depth = self.project(self.box_corners(box))
        if depth.max() < 0.5:
            return False
        vis = depth > 0.5
        h, w = self.image_size()
        u, v = uv[vis, 0], uv[vis, 1]
        return bool((u.max() > -margin) and (u.min() < w + margin) and
                    (v.max() > -margin) and (v.min() < h + margin))

    def image_size(self):
        if self._size is None:
            self._size = self.image().shape[:2]
        return self._size

    def labels(self, classes=CLASSES, max_truncation=0.9):
        """Ground truth, converted into the LIDAR frame.

        Two conversions that are easy to get subtly wrong: the label's `y` is the
        bottom of the box so the centre is `y - h/2`, and the camera yaw becomes
        a lidar heading as `-ry - pi/2`. Both are checked by the figure itself —
        a wrong one shows up as boxes that miss the cars they are labelling.
        """
        import bcdl

        if not os.path.exists(self._label_path):
            return []
        inv = np.linalg.inv(self.R0 @ self.Tr)
        out = []
        for line in open(self._label_path):
            f = line.split()
            if len(f) < 15 or f[0] not in classes:
                continue
            trunc = float(f[1])
            if trunc > max_truncation:
                continue
            h, w, l = float(f[8]), float(f[9]), float(f[10])
            x, y, z, ry = float(f[11]), float(f[12]), float(f[13]), float(f[14])
            centre_cam = np.array([x, y - h / 2.0, z, 1.0])
            cx, cy, cz = (inv @ centre_cam)[:3]
            # A real Box3D, not a look-alike: the labels go through the same
            # BEV IoU and the same drawing code as the predictions, and anything
            # that only quacks like one would quietly diverge from them.
            g = bcdl.Box3D()
            g.x, g.y, g.z = float(cx), float(cy), float(cz)
            g.dx, g.dy, g.dz = l, w, h
            g.heading = -ry - np.pi / 2.0
            g.label, g.score = classes.index(f[0]), 1.0
            out.append(g)
        return out


def match(preds, gts, iou_by_class=(0.7, 0.5, 0.5), bev_iou=None):
    """Greedy match by BEV IoU, highest-scoring prediction first.

    The thresholds are KITTI's own — 0.7 for cars, 0.5 for pedestrians and
    cyclists — so "matched" here means what it means in the benchmark, not
    whatever looked lenient enough to make the figure flattering.

    Returns (pred_state, gt_state): for each prediction, the index of the ground
    truth it took or -1; for each label, the prediction that took it or -1.
    """
    if bev_iou is None:
        import bcdl
        bev_iou = bcdl.bev_iou
    pred_hit = [-1] * len(preds)
    gt_hit = [-1] * len(gts)
    order = sorted(range(len(preds)), key=lambda i: -preds[i].score)
    for pi in order:
        p = preds[pi]
        best, best_iou = -1, 0.0
        for gi, g in enumerate(gts):
            if gt_hit[gi] >= 0 or g.label != p.label:
                continue
            iou = bev_iou(p, g)
            if iou > best_iou:
                best, best_iou = gi, iou
        thr = iou_by_class[p.label % len(iou_by_class)]
        if best >= 0 and best_iou >= thr:
            pred_hit[pi] = best
            gt_hit[best] = pi
    return pred_hit, gt_hit
