#!/usr/bin/env python3
"""
manager_node.py  ─  중앙제어 노드
═══════════════════════════════════════════════════════════════

[시나리오 흐름]
  STANDBY → SWEEP → SORTING ↔ PAUSED → DONE → STANDBY

[토픽 / 액션]
  구독  /recycle/command              SortCommand
  구독  /recycle/ui/command           SortCommand
  구독  /recycle/vision/detections    Detections2D
  발행  /recycle/response             SystemStatus
  액션  /recycle/exec/pick_place      PickPlace  (ActionClient)
"""

import queue
import threading
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Empty, String

from manager_node.bin_status import BinStatus
from manager_node.constants import (
    BIN_CAPACITY, EXCLUDE_BITS, LABEL_TO_BIN,
    PRIORITY_DEFAULT, Phase, SystemState,
)
from manager_node.exec_client import ExecActionClient

try:
    from recycle_interfaces.msg import Detections2D, SortCommand, SystemStatus
    INTERFACES_AVAILABLE = True
except ImportError:
    INTERFACES_AVAILABLE = False


class ManagerNode(Node):

    def __init__(self):
        super().__init__("manager_node")

        self.declare_parameter("status_rate",    2.0)
        self.declare_parameter("action_timeout", 120.0)
        self._status_rate    = self.get_parameter("status_rate").value
        self._action_timeout = self.get_parameter("action_timeout").value

        # ── 내부 상태 ────────────────────────────────────────
        self._lock             = threading.Lock()
        self._phase            = Phase.STANDBY
        self._state            = SystemState.IDLE
        self._priority_order   : list[str] = []
        self._exclude_mask     = 0
        self._current_label    = ""
        self._current_bin      = ""
        self._current_phase_fb = ""
        self._current_progress = 0.0
        self._counters         = {"total": 0, "plastic": 0, "can": 0, "paper": 0, "trash": 0}
        self._last_message     = "System initialized."
        self.bin_status        = BinStatus()

        self._det_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._det_seq   = 0
        self._pause_event = threading.Event()
        self._pause_event.set()

        # ── 콜백 그룹 ────────────────────────────────────────
        self._cb_sub   = ReentrantCallbackGroup()
        self._cb_timer = MutuallyExclusiveCallbackGroup()

        # ── 구독 / 발행 ──────────────────────────────────────
        if INTERFACES_AVAILABLE:
            self.create_subscription(SortCommand,  "/recycle/command",
                self._cmd_cb, 10, callback_group=self._cb_sub)
            self.create_subscription(SortCommand,  "/recycle/ui/command",
                self._cmd_cb, 10, callback_group=self._cb_sub)
            self.create_subscription(Detections2D, "/recycle/vision/detections",
                self._vision_cb, 10, callback_group=self._cb_sub)
            self._status_pub = self.create_publisher(SystemStatus, "/recycle/response", 10)
        else:
            self.create_subscription(String, "/recycle/command",
                lambda m: self.get_logger().info(f"[fallback CMD] {m.data}"), 10)
            self._status_pub = self.create_publisher(String, "/recycle/response", 10)

        self._vision_reset_pub = self.create_publisher(Empty, "/recycle/vision/reset", 10)

        # ── 액션 클라이언트 ──────────────────────────────────
        self._exec = ExecActionClient(
            self, timeout=self._action_timeout,
            feedback_cb=self._on_feedback,
            cb_group=self._cb_sub if INTERFACES_AVAILABLE else None,
        )

        self.create_timer(1.0 / self._status_rate, self._publish_status,
                          callback_group=self._cb_timer)

        threading.Thread(target=self._main_worker, daemon=True).start()
        self.get_logger().info("ManagerNode ready.")

    # ═══════════════════════════════════════════════════════
    #  명령 콜백
    # ═══════════════════════════════════════════════════════

    def _cmd_cb(self, msg):
        cmd  = msg.cmd.upper()
        mode = msg.mode.lower() if msg.mode else ""
        self.get_logger().info(
            f"[CMD] cmd={cmd} mode={mode} priority={list(msg.priority_order)} "
            f"exclude={msg.exclude_mask:#04x} raw='{msg.raw_text}'"
        )
        with self._lock:
            if cmd == "SWEEP" and self._phase == Phase.STANDBY:
                self._set_phase(Phase.SWEEP, SystemState.RUNNING, "Starting sweep.")

            elif cmd == "START":
                if mode in ("sorting", ""):
                    if msg.priority_order:
                        self._priority_order = list(msg.priority_order)
                    if msg.exclude_mask:
                        self._exclude_mask = msg.exclude_mask
                    self._set_phase(Phase.SORTING, SystemState.RUNNING, "Starting sorting.")
                elif mode == "standby":
                    self._set_phase(Phase.STANDBY, SystemState.IDLE, "Returning to standby.")

            elif cmd == "PAUSE" and self._phase not in (Phase.STANDBY, Phase.DONE, Phase.PAUSED):
                self._set_phase(Phase.PAUSED, SystemState.PAUSED,
                                "Paused. Will stop after current task.", event_set=False)

            elif cmd == "RESUME" and self._phase == Phase.PAUSED:
                self._set_phase(Phase.SORTING, SystemState.RUNNING, "Resumed sorting.")

            elif cmd == "STOP":
                self._set_phase(Phase.DONE, SystemState.STOPPED, "Stopping after current task.")

            elif cmd == "SET_POLICY":
                if msg.priority_order:
                    self._priority_order = list(msg.priority_order)
                self._exclude_mask = msg.exclude_mask
                self._last_message = (
                    f"Policy updated: priority_order={self._priority_order} "
                    f"exclude_mask={self._exclude_mask:#04x}"
                )

    def _set_phase(self, phase: Phase, state: SystemState, message: str, event_set: bool = True):
        """phase / state / message 일괄 갱신 + pause_event 제어 (lock 내부에서 호출)."""
        self._phase        = phase
        self._state        = state
        self._last_message = message
        if event_set:
            self._pause_event.set()
        else:
            self._pause_event.clear()

    # ═══════════════════════════════════════════════════════
    #  비전 콜백
    # ═══════════════════════════════════════════════════════

    def _vision_cb(self, msg):
        if self._phase != Phase.SORTING:
            return

        self.get_logger().info(
            f"[VISION_CB] SORTING: received labels={[d.label for d in msg.detections]}"
        )
        for det in msg.detections:
            bin_id = LABEL_TO_BIN.get(det.label.lower())
            if not bin_id:
                self.get_logger().warn(f"[VISION_CB] unknown label '{det.label}' – skip.")
                continue
            cat = bin_id.replace("BIN_", "")
            if self._exclude_mask & EXCLUDE_BITS.get(cat, 0):
                continue
            if self.bin_status.is_full(bin_id):
                self.get_logger().warn(f"[VISION] {bin_id} FULL, skip {det.label}")
                continue
            prio = self._calc_priority(cat)
            with self._lock:
                self._det_seq += 1
                self._det_queue.put_nowait((prio, self._det_seq, det))

    def _calc_priority(self, cat: str) -> int:
        if cat in self._priority_order:
            return self._priority_order.index(cat)
        return len(self._priority_order) + PRIORITY_DEFAULT.get(cat, 99)

    # ═══════════════════════════════════════════════════════
    #  메인 워커 스레드
    # ═══════════════════════════════════════════════════════

    def _main_worker(self):
        self.get_logger().info("[WORKER] Main worker started.")
        while rclpy.ok():
            phase = self._phase
            if phase == Phase.STANDBY:
                time.sleep(0.3)
            elif phase == Phase.SWEEP:
                self._run_sweep()
            elif phase == Phase.SORTING:
                self._run_sorting_step()
            elif phase == Phase.PAUSED:
                self.get_logger().info("[WORKER] Paused – waiting for RESUME.")
                self._pause_event.wait()
            elif phase == Phase.DONE:
                self._exec.send_exec_command("CLEAN_DESK")
                self._exec.send_exec_command("HOME")
                with self._lock:
                    self._set_phase(Phase.STANDBY, SystemState.STOPPED,
                                    "All done. Robot at home position.")
                time.sleep(1.0)

    def _run_sweep(self):
        self.get_logger().info("[PHASE] SWEEP")
        with self._lock:
            self._current_phase_fb = "SWEEP"
            self._last_message     = "Sweeping trash to separate items..."

        success = self._exec.send_exec_command("SWEEP")

        with self._lock:
            self._current_phase_fb = ""
            self._last_message = (
                "Sweep complete. Say '분류해' to start sorting."
                if success else "Sweep failed."
            )
            self._phase = Phase.STANDBY

    def _run_sorting_step(self):
        if not self._pause_event.is_set():
            return

        self.get_logger().info("[SORT] Moving to J_WORK for vision scan...")
        if not self._exec.send_exec_command("GOTO_WORK"):
            self.get_logger().warn("[SORT] GOTO_WORK failed – skipping scan cycle.")
            return

        # 로봇이 J_WORK에 완전히 정착할 때까지 대기
        time.sleep(1.0)

        # 이전 스캔 결과 버리기 → 리셋 신호 발행
        while not self._det_queue.empty():
            try: self._det_queue.get_nowait()
            except queue.Empty: break
        self._vision_reset_pub.publish(Empty())

        try:
            _, _, det = self._det_queue.get(timeout=2.0)
        except queue.Empty:
            self.get_logger().info("[SORT] No objects detected. Sorting complete.")
            with self._lock:
                self._phase = Phase.DONE
            return

        bin_id = LABEL_TO_BIN.get(det.label.lower())
        if bin_id is None or self.bin_status.is_full(bin_id):
            self.get_logger().warn(f"[SORT] skip '{det.label}' (unknown or full)")
            return

        with self._lock:
            self._current_label    = det.label
            self._current_bin      = bin_id
            self._current_phase_fb = "APPROACH"
            self._current_progress = 0.0
            self._last_message     = f"Sorting: {det.label} → {bin_id}"

        success = self._exec.send_pick_place(det, bin_id)

        with self._lock:
            cat = bin_id.replace("BIN_", "").lower()
            if success:
                self.bin_status.add(bin_id)
                self._counters["total"] += 1
                if cat in self._counters:
                    self._counters[cat] += 1
                remain = self.bin_status.remaining(bin_id)
                full_warn = " ⚠ BIN ALMOST FULL" if remain <= 3 else ""
                self._last_message = (
                    f"Placed {det.label} → {bin_id}  "
                    f"[{self.bin_status.count(bin_id)}/{BIN_CAPACITY[bin_id]}]"
                    f"{full_warn}"
                )
            else:
                self._last_message = f"Failed: {det.label}."
            self._current_label = self._current_bin = self._current_phase_fb = ""
            self._current_progress = 0.0

    # ═══════════════════════════════════════════════════════
    #  피드백 / 상태 발행
    # ═══════════════════════════════════════════════════════

    def _on_feedback(self, phase: str, progress: float):
        with self._lock:
            self._current_phase_fb = phase
            self._current_progress = progress

    def _publish_status(self):
        try:
            msg = SystemStatus()
            msg.stamp             = self.get_clock().now().to_msg()
            msg.state             = self._state.value
            msg.mode              = self._phase.value
            msg.priority_order    = self._priority_order
            msg.exclude_mask      = self._exclude_mask
            msg.processed_total   = self._counters["total"]
            msg.processed_plastic = self._counters["plastic"]
            msg.processed_can     = self._counters["can"]
            msg.processed_paper   = self._counters["paper"]
            msg.processed_trash   = self._counters["trash"]
            msg.last_message      = self._last_message
            msg.progress          = self._current_progress
            self._status_pub.publish(msg)
        except Exception:
            s = String()
            s.data = (f"state={self._state.value} phase={self._phase.value} "
                      f"total={self._counters['total']} msg={self._last_message}")
            self._status_pub.publish(s)


def main(args=None):
    rclpy.init(args=args)
    node = ManagerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
