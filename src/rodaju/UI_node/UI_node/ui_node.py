#!/usr/bin/env python3
"""
ui_node.py  ─  웹 대시보드 UI 노드
웹 브라우저에서 http://localhost:5000 으로 접속
"""

import os
import threading
import time
import json as _json

import rclpy
from rclpy.node import Node
from flask import Flask, Response, request, jsonify, send_from_directory
from ament_index_python.packages import get_package_share_directory

try:
    from recycle_interfaces.msg import SortCommand, SystemStatus
    INTERFACES_AVAILABLE = True
except ImportError:
    from std_msgs.msg import String
    INTERFACES_AVAILABLE = False

try:
    from sensor_msgs.msg import CompressedImage as _CompressedImage
    SENSOR_MSGS_AVAILABLE = True
except ImportError:
    SENSOR_MSGS_AVAILABLE = False

# 대시보드 모달용 voice 로그 수신
from std_msgs.msg import String as _StrMsg


# ═══════════════════════════════════════════════════════════════
#  공유 상태
# ═══════════════════════════════════════════════════════════════

class UIState:
    def __init__(self):
        self.state             = "IDLE"
        self.phase             = "STANDBY"
        self.priority_order: list[str] = []
        self.exclude_mask      = 0
        self.last_message      = "System ready."
        self.progress          = 0.0

        # 현재 처리 아이템 (last_message 파싱 보조)
        self.current_label     = ""
        self.current_bin       = ""
        self.current_robot_phase = ""

        # 카운터
        self.total    = 0
        self.plastic  = 0
        self.can      = 0
        self.paper    = 0
        self.trash    = 0

        # 분류함 용량 (처리된 개수로 역산)
        self.bin_counts: dict[str, int] = {
            "BIN_PLASTIC": 0, "BIN_CAN": 0, "BIN_PAPER": 0, "BIN_TRASH": 0}

        self.log_lines: list[str] = []
        self.lock = threading.Lock()

        # YOLO 프리뷰 최신 JPEG 바이트 (MJPEG 스트림용)
        self.latest_preview: bytes = b""
        self.preview_lock = threading.Lock()

    def update(self, msg):
        with self.lock:
            self.state    = msg.state
            self.phase    = msg.mode       # manager_node 는 mode 필드에 phase 값 넣음
            self.priority_order = list(msg.priority_order)
            self.exclude_mask = msg.exclude_mask
            self.last_message = msg.last_message
            self.progress = msg.progress
            self.total   = msg.processed_total
            self.plastic = msg.processed_plastic
            self.can     = msg.processed_can
            self.paper   = msg.processed_paper
            self.trash   = msg.processed_trash

            # bin_counts 동기화
            self.bin_counts["BIN_PLASTIC"] = self.plastic
            self.bin_counts["BIN_CAN"]     = self.can
            self.bin_counts["BIN_PAPER"]   = self.paper
            self.bin_counts["BIN_TRASH"]   = self.trash

            # last_message 에서 현재 아이템 파싱
            # 예: "Sorting: plastic → BIN_PLASTIC"
            import re
            m = re.search(r"Sorting:\s*(\w+)\s*→\s*(BIN_\w+)", msg.last_message)
            if m:
                self.current_label = m.group(1)
                self.current_bin   = m.group(2)
            elif "Placed" in msg.last_message or "Failed" in msg.last_message:
                self.current_label = ""
                self.current_bin   = ""
            self.current_robot_phase = self.phase

    def add_log(self, text: str):
        with self.lock:
            ts = time.strftime("%H:%M:%S")
            self.log_lines.append(f"[{ts}] {text}")
            if len(self.log_lines) > 300:
                self.log_lines = self.log_lines[-300:]


# ═══════════════════════════════════════════════════════════════
#  ROS 노드
# ═══════════════════════════════════════════════════════════════

class UINode(Node):
    def __init__(self, ui_state: UIState):
        super().__init__("ui_node")
        self._ui = ui_state

        if INTERFACES_AVAILABLE:
            self.create_subscription(SystemStatus, "/recycle/response",
                self._status_cb, 10)
            self._cmd_pub = self.create_publisher(SortCommand, "/recycle/ui/command", 10)
        else:
            self.create_subscription(String, "/recycle/response",
                lambda m: self._ui.add_log(m.data), 10)
            self._cmd_pub = self.create_publisher(String, "/recycle/ui/command", 10)

        # YOLO 프리뷰 구독
        if SENSOR_MSGS_AVAILABLE:
            self.create_subscription(
                _CompressedImage, "/recycle/vision/preview",
                self._preview_cb, 1)

        # voice_command_node 로그 → 대시보드 모달 트리거
        self.create_subscription(
            _StrMsg, "/recycle/voice/log",
            lambda msg: self._ui.add_log(msg.data), 10)

    def _status_cb(self, msg):
        self._ui.update(msg)
        self._ui.add_log(
            f"[STATUS] {msg.state} | {msg.mode} | "
            f"total={msg.processed_total} | {msg.last_message[:60]}"
        )

    def _preview_cb(self, msg):
        with self._ui.preview_lock:
            self._ui.latest_preview = bytes(msg.data)

    def send_cmd(self, cmd: str, mode: str = "", priority_order: list = None,
                 exclude_mask: int = 0, raw: str = ""):
        try:
            msg = SortCommand()
            msg.stamp          = self.get_clock().now().to_msg()
            msg.cmd            = cmd
            msg.mode           = mode
            msg.priority_order = priority_order or []
            msg.exclude_mask   = exclude_mask
            msg.raw_text       = raw or f"UI:{cmd}"
            self._cmd_pub.publish(msg)
            self._ui.add_log(f"[CMD] {cmd} mode={mode} priority_order={msg.priority_order}")
        except Exception as e:
            self.get_logger().error(f"Publish error: {e}")


# ═══════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════

def run_web_server(ui: UIState, node: UINode, host: str = "0.0.0.0", port: int = 5000):
    try:
        html_dir = os.path.join(get_package_share_directory("UI_node"), "resource")
    except Exception:
        html_dir = os.path.join(os.path.dirname(__file__), "..", "resource")
    app = Flask(__name__)

    @app.route("/")
    def index():
        return send_from_directory(html_dir, "dashboard.html")

    @app.route("/stream")
    def stream():
        def event_gen():
            while True:
                with ui.lock:
                    data = {
                        "state"         : ui.state,
                        "phase"         : ui.phase,
                        "priority_order": ui.priority_order,
                        "exclude_mask"  : ui.exclude_mask,
                        "last_message"  : ui.last_message,
                        "progress"      : ui.progress,
                        "total"         : ui.total,
                        "plastic"       : ui.plastic,
                        "can"           : ui.can,
                        "paper"         : ui.paper,
                        "bin_counts"    : ui.bin_counts,
                        "logs"          : ui.log_lines[-30:],
                    }
                yield f"data: {_json.dumps(data)}\n\n"
                time.sleep(0.5)
        return Response(event_gen(), mimetype="text/event-stream")

    @app.route("/cmd", methods=["POST"])
    def cmd():
        body = request.get_json(force=True)
        node.send_cmd(
            cmd           = body.get("cmd", ""),
            mode          = body.get("mode", ""),
            priority_order= body.get("priority_order", []),
            exclude_mask  = body.get("exclude_mask", 0),
            raw           = body.get("raw", ""),
        )
        return jsonify({"ok": True})

    @app.route("/video_feed")
    def video_feed():
        """MJPEG 스트림: YOLO 시각화 프리뷰"""
        _BOUNDARY = b"--frame\r\n"
        _NO_FRAME = (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"  # 빈 프레임 → 클라이언트는 이전 이미지 유지
        )

        def mjpeg_gen():
            while True:
                with ui.preview_lock:
                    frame = ui.latest_preview
                if frame:
                    yield (
                        _BOUNDARY
                        + b"Content-Type: image/jpeg\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                time.sleep(0.05)   # ~20 fps

        return Response(
            mjpeg_gen(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.run(host=host, port=port, threaded=True)


def main(args=None):
    rclpy.init(args=args)
    ui_state = UIState()
    ui_node  = UINode(ui_state)

    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(ui_node), daemon=True)
    spin_thread.start()

    web_thread = threading.Thread(
        target=run_web_server, args=(ui_state, ui_node), daemon=True)
    web_thread.start()

    print("[UI] Web dashboard: http://localhost:5000")

    try:
        web_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        ui_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()