#!/usr/bin/env python3
"""
vision_node.py  ─  카메라 인식 노드
═══════════════════════════════════════════════════════════════

[감지 모드]
  TRASH_MODE  ─ 개별 쓰레기 감지 + 분류 (SORTING 페이즈)
  ※ 쓰레기 봉투는 고정 좌표 사용 (디텍션 없음)

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

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import Empty
from cv_bridge import CvBridge

from vision_node.depth_utils import pixel_to_3d, obb_to_3d

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
#  VisionNode
# ═══════════════════════════════════════════════════════════════

class VisionNode(Node):

    def __init__(self):
        super().__init__("vision_node")

        # ── 파라미터 ────────────────────────────────────────
        self.declare_parameter("publish_rate",    10.0)
        self.declare_parameter("conf_threshold",   0.70)
        self.declare_parameter("depth_scale",      0.001)
        self.declare_parameter("depth_roi_radius",    3)

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

        self._BUCKET     = 50
        self._published: set[tuple] = set()
        self._scan_frames_remaining = 0

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
        self.create_subscription(Empty, "/recycle/vision/reset",
            self._reset_cb, 10)

        # ── 발행 ────────────────────────────────────────────
        if INTERFACES_AVAILABLE:
            self._pub = self.create_publisher(Detections2D, "/recycle/vision/detections", 10)
        else:
            self._pub = self.create_publisher(String, "/recycle/vision/detections", 10)

        self._preview_pub = self.create_publisher(
            CompressedImage, "/recycle/vision/preview", 1)

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

    def _reset_cb(self, _msg):
        """sweep 완료 후 manager_node가 발행하는 리셋 신호: one-shot 스캔 활성화."""
        self._published.clear()
        self._scan_frames_remaining = 10
        self.get_logger().info("[VISION] One-shot scan activated (10 frames / 1s).")

    def _info_cb(self, msg: CameraInfo):
        if self._intrinsics is None:
            self._intrinsics = {
                "fx": msg.k[0], "fy": msg.k[4],
                "ppx": msg.k[2], "ppy": msg.k[5],
            }
            self.get_logger().info(f"Intrinsics: {self._intrinsics}")

    # ═══════════════════════════════════════════════════════
    #  감지 + 발행 (타이머)
    # ═══════════════════════════════════════════════════════

    def _detect_and_publish(self):
        import cv2
        if self._color_frame is None:
            return

        if self._scan_frames_remaining <= 0:
            return
        self._scan_frames_remaining -= 1
        self.get_logger().info(
            f"[VISION] Scanning... ({self._scan_frames_remaining} frames left)"
        )

        raws = self._run_yolo(self._color_frame)
        dets = []

        current_keys: set[tuple] = set()
        for raw in raws:
            key = (raw["label"], raw["cx"] // self._BUCKET, raw["cy"] // self._BUCKET)
            current_keys.add(key)
            if key in self._published:
                continue
            det = self._build_detection(raw)
            if det:
                dets.append(det)
                self._published.add(key)

        self._published &= current_keys

        # 바운딩박스 시각화
        vis = self._color_frame.copy()
        for raw in raws:
            label_text = f"{raw['label']} {raw['confidence']:.2f}"
            if raw.get("points") is not None:
                cv2.polylines(vis, [raw["points"]], isClosed=True, color=(0, 255, 0), thickness=2)
                x, y = int(raw["points"][0][0]), int(raw["points"][0][1])
            else:
                x1, y1 = raw["x"], raw["y"]
                x2, y2 = x1 + raw["w"], y1 + raw["h"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                x, y = x1, y1
            cv2.putText(vis, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if raw.get("grasp_px"):
                cv2.circle(vis, raw["grasp_px"], 8, (0, 0, 255), -1)
                cv2.circle(vis, raw["grasp_px"], 10, (255, 255, 255), 2)
        # 시각화 프레임 → ROS CompressedImage 발행
        try:
            import cv2 as _cv2
            ok, buf = _cv2.imencode(".jpg", vis, [_cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                cmsg = CompressedImage()
                cmsg.header.stamp = self.get_clock().now().to_msg()
                cmsg.format = "jpeg"
                cmsg.data   = buf.tobytes()
                self._preview_pub.publish(cmsg)
        except Exception as _e:
            self.get_logger().debug(f"Preview publish error: {_e}")

        if not dets:
            if raws:
                self.get_logger().debug(
                    f"[VISION] {len(raws)} raw dets skipped (already published)"
                )
            return

        try:
            msg            = Detections2D()
            msg.stamp      = self.get_clock().now().to_msg()
            msg.frame_id   = "camera_color_optical_frame"
            msg.detections = dets
            self._pub.publish(msg)
            self.get_logger().debug(f"[VISION] Published {len(dets)} detections")
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
                    label     = res.names[int(cls)]
                    points    = None
                    angle_deg = 0.0
                    if is_obb:
                        if hasattr(boxes_src, "xyxyxyxy"):
                            points = boxes_src.xyxyxyxy[i].cpu().numpy().astype(np.int32)
                        if hasattr(boxes_src, "xywhr"):
                            import math
                            angle_deg = float(boxes_src.xywhr[i][4].item()) * 180.0 / math.pi
                    dets.append({
                        "label"     : label,
                        "confidence": float(score),
                        "x": x1, "y": y1,
                        "w": x2 - x1, "h": y2 - y1,
                        "cx": (x1 + x2) // 2,
                        "cy": (y1 + y2) // 2,
                        "points"    : points,
                        "angle_deg" : angle_deg,
                    })
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

            if raw.get("points") is not None:
                x_m, y_m, z_m, gx, gy = obb_to_3d(
                    raw["points"], self._depth_frame, self._intrinsics,
                    depth_scale=self._depth_scl, logger=self.get_logger(),
                )
                raw["grasp_px"] = (int(gx), int(gy))
            else:
                x_m, y_m, z_m = pixel_to_3d(
                    raw["cx"], raw["cy"], self._depth_frame, self._intrinsics,
                    depth_scale=self._depth_scl, roi_radius=self._depth_roi,
                    logger=self.get_logger(),
                )

            det.has_3d = z_m > 0.01
            det.x_m, det.y_m, det.z_m = float(x_m), float(y_m), float(z_m)
            det.angle_deg = float(raw.get("angle_deg", 0.0))
            return det
        except Exception as e:
            self.get_logger().warn(f"Build detection error: {e}")
            return None


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
