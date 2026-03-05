#!/usr/bin/env python3
"""
vision_node.py  ─  카메라 인식 노드
═══════════════════════════════════════════════════════════════

[감지 모드]
  BAG_MODE    ─ 쓰레기 봉투 감지 (BAG_PICKUP 페이즈)
  TRASH_MODE  ─ 개별 쓰레기 감지 + 분류 (SORTING 페이즈)

[처리 흐름]
  RGB 이미지 수신
    → YOLO 추론
    → 중심 픽셀에서 depth 값 조회
    → 카메라 내부 파라미터로 3D 좌표 변환
    → /recycle/vision/detections 발행

[구독]
  /camera/camera/color/image_raw
  /camera/camera/aligned_depth_to_color/image_raw
  /camera/camera/color/camera_info

[발행]
  /recycle/vision/detections   (Detections2D)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

try:
    from recycle_interfaces.msg import Detection2D, Detections2D
    INTERFACES_AVAILABLE = True
except ImportError:
    from std_msgs.msg import String
    INTERFACES_AVAILABLE = False

try:
    from vision_node.yolo import YoloModel
    YOLO_AVAILABLE = True
except Exception as _yolo_err:
    print(f"[WARN] YoloModel import failed: {_yolo_err}")
    YOLO_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
#  레이블 매핑
# ═══════════════════════════════════════════════════════════════

# YOLO 클래스 → 분류 레이블 정규화
LABEL_NORM: dict[str, str] = {
    # 500ml 생수 페트병
    "plastic"       : "pet",
    "plastic_bottle": "pet",
    "bottle"        : "pet",
    "pet_bottle"    : "pet",
    "water_bottle"  : "pet",
    # 캔
    "can"           : "can",
    "tin_can"       : "can",
    "aluminum_can"  : "can",
    "metal"         : "can",
    # 종이컵
    "paper"         : "paper_cup",
    "paper_cup"     : "paper_cup",
    "cup"           : "paper_cup",
    # 봉투 클래스
    "trash_bag"     : "trash_bag",
    "bag"           : "trash_bag",
    "plastic_bag"   : "trash_bag",
}

# 봉투 레이블 (BAG 모드 필터에 사용)
BAG_LABELS = {"trash_bag", "bag", "plastic_bag"}


# ═══════════════════════════════════════════════════════════════
#  VisionNode
# ═══════════════════════════════════════════════════════════════

class VisionNode(Node):

    def __init__(self):
        super().__init__("vision_node")

        # ── 파라미터 ────────────────────────────────────────
        self.declare_parameter("publish_rate",    10.0)
        self.declare_parameter("conf_threshold",   0.25)
        self.declare_parameter("depth_scale",      0.001)  # mm → m
        self.declare_parameter("depth_roi_radius",    3)   # 중심 주변 평균 반경

        self._rate       = self.get_parameter("publish_rate").value
        self._conf       = self.get_parameter("conf_threshold").value
        self._depth_scl  = self.get_parameter("depth_scale").value
        self._depth_roi  = self.get_parameter("depth_roi_radius").value

        # ── 내부 상태 ────────────────────────────────────────
        self._bridge      = CvBridge()
        self._color_frame = None
        self._depth_frame = None
        self._intrinsics  = None
        self._det_id_cnt  = 0
        self._detect_mode = "TRASH"   # "BAG" | "TRASH"  (manager_node 요청으로 변경 가능)

        # ── YOLO ────────────────────────────────────────────
        if YOLO_AVAILABLE:
            self._yolo = YoloModel()
            self.get_logger().info("YOLO model loaded.")
        else:
            self._yolo = None
            self.get_logger().warn("YoloModel unavailable – dummy mode.")

        # ── 구독 ────────────────────────────────────────────
        qos = QoSProfile(depth=10)
        self.create_subscription(Image,      "/camera/camera/color/image_raw",
            self._color_cb, qos)
        self.create_subscription(Image,      "/camera/camera/aligned_depth_to_color/image_raw",
            self._depth_cb, qos)
        self.create_subscription(CameraInfo, "/camera/camera/color/camera_info",
            self._info_cb, qos)

        # 감지 모드 제어 토픽 (manager_node 에서 발행)
        from std_msgs.msg import String as Str
        self.create_subscription(Str, "/recycle/vision/mode",
            self._mode_cb, 10)

        # ── 발행 ────────────────────────────────────────────
        if INTERFACES_AVAILABLE:
            self._pub = self.create_publisher(Detections2D, "/recycle/vision/detections", 10)
        else:
            self._pub = self.create_publisher(String, "/recycle/vision/detections", 10)

        # ── 타이머 ──────────────────────────────────────────
        self.create_timer(1.0 / self._rate, self._detect_and_publish)

        self.get_logger().info("VisionNode ready.")

    # ═══════════════════════════════════════════════════════
    #  구독 콜백
    # ═══════════════════════════════════════════════════════

    def _color_cb(self, msg):
        self._color_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _depth_cb(self, msg):
        self._depth_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def _info_cb(self, msg: CameraInfo):
        if self._intrinsics is None:
            self._intrinsics = {
                "fx": msg.k[0], "fy": msg.k[4],
                "ppx": msg.k[2], "ppy": msg.k[5],
            }
            self.get_logger().info(f"Intrinsics: {self._intrinsics}")

    def _mode_cb(self, msg):
        """manager_node 에서 감지 모드 변경."""
        mode = msg.data.upper()
        if mode in ("BAG", "TRASH"):
            old = self._detect_mode
            self._detect_mode = mode
            self.get_logger().info(f"[MODE] {old} → {mode}")

    # ═══════════════════════════════════════════════════════
    #  감지 + 발행 (타이머)
    # ═══════════════════════════════════════════════════════

    def _detect_and_publish(self):
        import cv2
        if self._color_frame is None:
            return

        raws = self._run_yolo(self._color_frame)
        dets = []

        for raw in raws:
            is_bag = raw["label"] in ("trash_bag", "bag", "plastic_bag")

            # 모드별 필터
            if self._detect_mode == "BAG"   and not is_bag:
                continue
            if self._detect_mode == "TRASH" and is_bag:
                continue

            det = self._build_detection(raw)
            if det:
                dets.append(det)

        # 바운딩박스 시각화
        vis = self._color_frame.copy()
        for raw in raws:
            label_text = f"{raw['label']} {raw['confidence']:.2f}"
            if raw.get("points") is not None:
                # OBB: 회전된 다각형
                cv2.polylines(vis, [raw["points"]], isClosed=True, color=(0, 255, 0), thickness=2)
                x, y = int(raw["points"][0][0]), int(raw["points"][0][1])
            else:
                # 일반 bbox
                x1, y1 = raw["x"], raw["y"]
                x2, y2 = x1 + raw["w"], y1 + raw["h"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                x, y = x1, y1
            cv2.putText(vis, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("vision_node", vis)
        cv2.waitKey(1)

        # 감지 결과가 없으면 발행하지 않음
        # (빈 메시지가 manager_node 큐에 노이즈로 쌓이는 것 방지)
        if not dets:
            return

        try:
            msg            = Detections2D()
            msg.stamp      = self.get_clock().now().to_msg()
            msg.frame_id   = "camera_color_optical_frame"
            msg.detections = dets
            self._pub.publish(msg)
            self.get_logger().debug(
                f"[VISION] Published {len(dets)} detections (mode={self._detect_mode})"
            )
        except Exception:
            from std_msgs.msg import String as Str
            s      = Str()
            s.data = str([f"{d.label}@({d.cx},{d.cy})" for d in dets])
            self._pub.publish(s)

    # ═══════════════════════════════════════════════════════
    #  YOLO 추론
    # ═══════════════════════════════════════════════════════

    def _run_yolo(self, frame) -> list:
        if self._yolo is None:
            return []
        try:
            results = self._yolo.model([frame], verbose=False)
            dets = []
            for res in results:
                # OBB 모델 우선, 없으면 일반 boxes
                boxes_src = res.obb if (res.obb is not None and len(res.obb)) else res.boxes
                if boxes_src is None or len(boxes_src) == 0:
                    continue
                import numpy as np
                is_obb = boxes_src is res.obb
                for i, (box, score, cls) in enumerate(zip(
                    boxes_src.xyxy.tolist(),
                    boxes_src.conf.tolist(),
                    boxes_src.cls.tolist(),
                )):
                    if score < self._conf:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box]
                    raw_label = res.names[int(cls)]
                    label     = LABEL_NORM.get(raw_label.lower(), raw_label.lower())
                    # OBB면 4개 꼭짓점 저장
                    points = None
                    if is_obb and hasattr(boxes_src, "xyxyxyxy"):
                        points = boxes_src.xyxyxyxy[i].cpu().numpy().astype(np.int32)
                    dets.append({
                        "label"     : label,
                        "confidence": float(score),
                        "x": x1, "y": y1,
                        "w": x2 - x1, "h": y2 - y1,
                        "cx": (x1 + x2) // 2,
                        "cy": (y1 + y2) // 2,
                        "points"    : points,
                    })
            if dets:
                self.get_logger().info(f"[YOLO] {[(d['label'], round(d['confidence'],2)) for d in dets]}")
            return dets
        except Exception as e:
            self.get_logger().error(f"YOLO error: {e}")
            return []

    # ═══════════════════════════════════════════════════════
    #  Detection2D 빌드
    # ═══════════════════════════════════════════════════════

    def _build_detection(self, raw: dict):
        try:
            det            = Detection2D()
            self._det_id_cnt += 1
            det.id         = self._det_id_cnt
            det.label      = raw["label"]
            det.confidence = raw["confidence"]
            det.x, det.y   = raw["x"], raw["y"]
            det.w, det.h   = raw["w"], raw["h"]
            det.cx, det.cy = raw["cx"], raw["cy"]

            x_m, y_m, z_m = self._pixel_to_3d(raw["cx"], raw["cy"])
            det.has_3d = z_m > 0.01
            det.x_m, det.y_m, det.z_m = float(x_m), float(y_m), float(z_m)
            return det
        except Exception as e:
            self.get_logger().warn(f"Build detection error: {e}")
            return None

    # ═══════════════════════════════════════════════════════
    #  2D → 3D 변환 (ROI 평균 depth)
    # ═══════════════════════════════════════════════════════

    def _pixel_to_3d(self, cx: int, cy: int):
        if self._depth_frame is None or self._intrinsics is None:
            return 0.0, 0.0, 0.0
        try:
            import numpy as np
            h, w = self._depth_frame.shape[:2]
            r    = self._depth_roi
            x1   = max(0, cx - r); x2 = min(w, cx + r + 1)
            y1   = max(0, cy - r); y2 = min(h, cy + r + 1)
            roi  = self._depth_frame[y1:y2, x1:x2].astype(float)
            valid = roi[roi > 0]
            if valid.size == 0:
                return 0.0, 0.0, 0.0

            z_m = float(np.median(valid)) * self._depth_scl
            if z_m < 0.01:
                return 0.0, 0.0, 0.0

            fx, fy   = self._intrinsics["fx"],  self._intrinsics["fy"]
            ppx, ppy = self._intrinsics["ppx"], self._intrinsics["ppy"]
            x_m = (cx - ppx) * z_m / fx
            y_m = (cy - ppy) * z_m / fy
            return x_m, y_m, z_m
        except Exception as e:
            self.get_logger().warn(f"3D transform error: {e}")
            return 0.0, 0.0, 0.0


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
