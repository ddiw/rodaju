#!/usr/bin/env python3
"""
ui_node.py  ─  웹 대시보드 UI 노드
웹 브라우저에서 http://localhost:5000 으로 접속
"""

import threading

import rclpy
from rclpy.node import Node

from UI_node.ui_state import UIState
from UI_node.web_server import run_web_server

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

from std_msgs.msg import String as _StrMsg


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
