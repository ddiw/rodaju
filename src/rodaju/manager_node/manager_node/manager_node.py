#!/usr/bin/env python3
"""
manager_node.py  ─  중앙제어 노드
═══════════════════════════════════════════════════════════════

[시나리오 흐름]
  PHASE 1. STANDBY  ─ 초기 대기
  PHASE 2. SWEEP    ─ 주걱으로 쓰레기 훑어 이격
  PHASE 3. SORTING  ─ 음성/UI 명령으로 지정된 종류 순서대로 분류
  PHASE 4. PAUSED   ─ 일시정지 (현재 동작 완료 후 대기)
  PHASE 5. DONE     ─ 완료 후 홈 복귀

[토픽 / 액션]
  구독  /recycle/command              SortCommand
  구독  /recycle/ui/command           SortCommand
  구독  /recycle/vision/detections    Detections2D
  발행  /recycle/response             SystemStatus
  액션  /recycle/exec/pick_place      PickPlace  (ActionClient)
"""

import threading
import queue
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String, Empty

try:
    from recycle_interfaces.msg    import SortCommand, SystemStatus, Detections2D, Detection2D
    from recycle_interfaces.action import PickPlace
    INTERFACES_AVAILABLE = True
except ImportError:
    INTERFACES_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
#  상수 / 열거형
# ═══════════════════════════════════════════════════════════════

class Phase(Enum):
    STANDBY = "STANDBY"
    SWEEP   = "SWEEP"
    SORTING = "SORTING"
    PAUSED  = "PAUSED"
    DONE    = "DONE"

class SystemState(Enum):
    IDLE    = "IDLE"
    RUNNING = "RUNNING"
    PAUSED  = "PAUSED"
    STOPPED = "STOPPED"
    ERROR   = "ERROR"

LABEL_TO_BIN: dict[str, str] = {
    "pet"          : "BIN_PLASTIC",   # 500ml 생수 페트병
    "bottle"       : "BIN_PLASTIC",
    "plastic_bottle": "BIN_PLASTIC",
    "plastic"      : "BIN_PLASTIC",
    "water_bottle" : "BIN_PLASTIC",
    "can"          : "BIN_CAN",       # 캔
    "metal"        : "BIN_CAN",
    "aluminum"     : "BIN_CAN",
    "paper_cup"    : "BIN_PAPER",     # 종이컵
    "paper"        : "BIN_PAPER",
    "cup"          : "BIN_PAPER",
}

# 분류함 최대 용량 (개수 기준, 실제 프로젝트에서 조정)
BIN_CAPACITY: dict[str, int] = {
    "BIN_PLASTIC": 20,
    "BIN_CAN"    : 15,
    "BIN_PAPER"  : 25,
}

EXCLUDE_BITS: dict[str, int] = {"PLASTIC": 1, "CAN": 2, "PAPER": 4}

PRIORITY_DEFAULT: dict[str, int] = {
    "PLASTIC": 1, "CAN": 2, "PAPER": 3, "NONE": 99
}


# ═══════════════════════════════════════════════════════════════
#  분류함 용량 관리
# ═══════════════════════════════════════════════════════════════

class BinStatus:
    def __init__(self):
        self._counts: dict[str, int] = {k: 0 for k in BIN_CAPACITY}
        self._lock = threading.Lock()

    def add(self, bin_id: str):
        with self._lock:
            if bin_id in self._counts:
                self._counts[bin_id] += 1

    def count(self, bin_id: str) -> int:
        return self._counts.get(bin_id, 0)

    def remaining(self, bin_id: str) -> int:
        return max(0, BIN_CAPACITY.get(bin_id, 0) - self._counts.get(bin_id, 0))

    def percent_full(self, bin_id: str) -> float:
        cap = BIN_CAPACITY.get(bin_id, 1)
        return min(100.0, self._counts.get(bin_id, 0) / cap * 100.0)

    def is_full(self, bin_id: str) -> bool:
        return self.remaining(bin_id) <= 0

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._counts)


# ═══════════════════════════════════════════════════════════════
#  ManagerNode
# ═══════════════════════════════════════════════════════════════

class ManagerNode(Node):

    def __init__(self):
        super().__init__("manager_node")

        # ── 파라미터 ────────────────────────────────────────
        self.declare_parameter("status_rate",       2.0)
        self.declare_parameter("action_timeout",   40.0)

        self._status_rate    = self.get_parameter("status_rate").value
        self._action_timeout = self.get_parameter("action_timeout").value

        # ── 내부 상태 ────────────────────────────────────────
        self._lock         = threading.Lock()
        self._phase        = Phase.STANDBY
        self._state        = SystemState.IDLE
        self._priority_order: list[str] = []
        self._exclude_mask = 0
        self._target_label = "ALL"

        # 현재 처리 중인 아이템
        self._current_label    : str   = ""
        self._current_bin      : str   = ""
        self._current_phase_fb : str   = ""
        self._current_progress : float = 0.0

        # 처리 카운터
        self._counters = {"total": 0, "plastic": 0, "can": 0, "paper": 0, "trash": 0}
        self.bin_status = BinStatus()   # UI 노드에서도 접근
        self._last_message = "System initialized."

        # 감지 우선순위 큐
        self._det_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._det_seq   = 0

        # PAUSE/RESUME 동기화
        self._pause_event = threading.Event()
        self._pause_event.set()   # 초기 non-paused


        # ── 콜백 그룹 ────────────────────────────────────────
        self._cb_sub   = ReentrantCallbackGroup()
        self._cb_timer = MutuallyExclusiveCallbackGroup()

        # ── 구독 ────────────────────────────────────────────
        if INTERFACES_AVAILABLE:
            self.create_subscription(SortCommand,  "/recycle/command",
                self._cmd_cb, 10, callback_group=self._cb_sub)
            self.create_subscription(SortCommand,  "/recycle/ui/command",
                self._cmd_cb, 10, callback_group=self._cb_sub)
            self.create_subscription(Detections2D, "/recycle/vision/detections",
                self._vision_cb, 10, callback_group=self._cb_sub)
        else:
            self.create_subscription(
                String, "/recycle/command",
                lambda m: self.get_logger().info(f"[fallback CMD] {m.data}"), 10)

        # ── 발행 ────────────────────────────────────────────
        if INTERFACES_AVAILABLE:
            self._status_pub = self.create_publisher(SystemStatus, "/recycle/response", 10)
        else:
            self._status_pub = self.create_publisher(String, "/recycle/response", 10)
        self._vision_reset_pub = self.create_publisher(Empty, "/recycle/vision/reset", 10)

        # ── 액션 클라이언트 ──────────────────────────────────
        if INTERFACES_AVAILABLE:
            self._action_client = ActionClient(
                self, PickPlace, "/recycle/exec/pick_place",
                callback_group=self._cb_sub)
        else:
            self._action_client = None

        # ── 상태 발행 타이머 ─────────────────────────────────
        self.create_timer(1.0 / self._status_rate,
                          self._publish_status,
                          callback_group=self._cb_timer)

        # ── 메인 워커 스레드 ─────────────────────────────────
        self._worker = threading.Thread(target=self._main_worker, daemon=True)
        self._worker.start()

        self.get_logger().info("ManagerNode ready.")

    # ═══════════════════════════════════════════════════════
    #  명령 콜백  (voice_command_node / ui_node 에서 수신)
    # ═══════════════════════════════════════════════════════

    def _cmd_cb(self, msg):
        cmd  = msg.cmd.upper()
        mode = msg.mode.lower() if msg.mode else ""
        self.get_logger().info(
            f"[CMD] cmd={cmd} mode={mode} priority_order={list(msg.priority_order)} "
            f"exclude={msg.exclude_mask:#04x} raw='{msg.raw_text}'"
        )

        with self._lock:
            if cmd == "SWEEP":
                if self._phase == Phase.STANDBY:
                    self._phase = Phase.SWEEP
                    self._state = SystemState.RUNNING
                    self._last_message = "Starting sweep."
                    self._pause_event.set()

            elif cmd == "START":
                if mode in ("sorting", ""):
                    if msg.priority_order:
                        self._priority_order = list(msg.priority_order)
                    if msg.exclude_mask:
                        self._exclude_mask = msg.exclude_mask
                    self._phase = Phase.SORTING
                    self._state = SystemState.RUNNING
                    self._last_message = "Starting sorting."
                    self._pause_event.set()
                elif mode == "standby":
                    self._phase = Phase.STANDBY
                    self._state = SystemState.IDLE
                    self._last_message = "Returning to standby."
                    self._pause_event.set()

            elif cmd == "PAUSE":
                if self._phase not in (Phase.STANDBY, Phase.DONE, Phase.PAUSED):
                    self._phase = Phase.PAUSED
                    self._state = SystemState.PAUSED
                    self._pause_event.clear()
                    self._last_message = "Paused. Will stop after current task."

            elif cmd == "RESUME":
                if self._phase == Phase.PAUSED:
                    self._phase = Phase.SORTING
                    self._state = SystemState.RUNNING
                    self._pause_event.set()
                    self._last_message = "Resumed sorting."

            elif cmd == "STOP":
                self._phase = Phase.DONE
                self._state = SystemState.STOPPED
                self._pause_event.set()
                self._last_message = "Stopping after current task."

            elif cmd == "SET_POLICY":
                if msg.priority_order:
                    self._priority_order = list(msg.priority_order)
                self._exclude_mask = msg.exclude_mask
                self._last_message = (
                    f"Policy updated: priority_order={self._priority_order} "
                    f"exclude_mask={self._exclude_mask:#04x}"
                )

    # ═══════════════════════════════════════════════════════
    #  비전 콜백
    # ═══════════════════════════════════════════════════════

    def _vision_cb(self, msg):
        """SORTING 페이즈: 쓰레기 감지 결과를 우선순위 큐에 삽입."""
        if self._phase != Phase.SORTING:
            self.get_logger().debug(
                f"[VISION_CB] phase={self._phase.value}, skip {len(msg.detections)} dets"
            )
            return

        labels = [d.label for d in msg.detections]
        self.get_logger().info(f"[VISION_CB] SORTING: received labels={labels}")

        # SORTING 페이즈: 쓰레기 감지 큐 삽입 (vision_node가 신규 물체만 발행)
        for det in msg.detections:
            bin_id = LABEL_TO_BIN.get(det.label.lower())
            if not bin_id:
                self.get_logger().warn(
                    f"[VISION_CB] label '{det.label}' not in LABEL_TO_BIN – skip. "
                    f"Known keys: {list(LABEL_TO_BIN.keys())}"
                )
                continue

            cat = bin_id.replace("BIN_", "")

            # exclude_mask 필터
            if self._exclude_mask & EXCLUDE_BITS.get(cat, 0):
                continue

            # 만석 스킵
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
        base = len(self._priority_order)
        return base + PRIORITY_DEFAULT.get(cat, 99)


    # ═══════════════════════════════════════════════════════
    #  메인 워커 스레드 (시나리오 제어)
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
                self._send_exec_command("HOME")
                with self._lock:
                    self._state        = SystemState.STOPPED
                    self._last_message = "All done. Robot at home position."
                    self._phase        = Phase.STANDBY
                time.sleep(1.0)

    # ─────────────────────────────────────────────────────────
    #  PHASE 2: 훑기
    # ─────────────────────────────────────────────────────────

    def _run_sweep(self):
        self.get_logger().info("[PHASE] SWEEP")
        with self._lock:
            self._current_phase_fb = "SWEEP"
            self._last_message     = "Sweeping trash to separate items..."

        success = self._send_exec_command("SWEEP")

        with self._lock:
            self._current_phase_fb = ""
            self._last_message = (
                "Sweep complete. Say '분류해' to start sorting."
                if success else
                "Sweep failed. Say '분류해' to start sorting."
            )
            self._phase = Phase.STANDBY
        self.get_logger().info("[SWEEP] Complete. Waiting for sort command.")

    # ─────────────────────────────────────────────────────────
    #  PHASE 3: 분류 (1 아이템 단위)
    # ─────────────────────────────────────────────────────────

    def _run_sorting_step(self):
        # PAUSE 체크
        if not self._pause_event.is_set():
            return

        # 매 스텝: J_WORK로 이동 → 큐 비우기 → 비전 5초 스캔 → 감지 기다리기
        self.get_logger().info("[SORT] Moving to J_WORK for vision scan...")
        self._send_exec_command("GOTO_WORK")

        # 이전 스캔 결과 버리기
        while not self._det_queue.empty():
            try:
                self._det_queue.get_nowait()
            except queue.Empty:
                break

        self._vision_reset_pub.publish(Empty())
        self.get_logger().info("[SORT] Vision scan started (1s). Waiting for detections...")

        # 스캔 완료될 때까지 대기 (1초 + 여유 1초)
        try:
            prio, seq, det = self._det_queue.get(timeout=2.0)
        except queue.Empty:
            self.get_logger().info("[SORT] No objects detected. Sorting complete.")
            with self._lock:
                self._phase = Phase.DONE
            return

        bin_id = LABEL_TO_BIN.get(det.label.lower())
        if bin_id is None:
            self.get_logger().warn(f"[SORT] Unknown label '{det.label}' – skip.")
            return

        if self.bin_status.is_full(bin_id):
            self.get_logger().warn(f"[SORT] {bin_id} full – skip {det.label}")
            return

        with self._lock:
            self._current_label    = det.label
            self._current_bin      = bin_id
            self._current_phase_fb = "APPROACH"
            self._current_progress = 0.0
            self._last_message     = f"Sorting: {det.label} → {bin_id}"

        success = self._send_pick_place(det, bin_id)

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

            self._current_label    = ""
            self._current_bin      = ""
            self._current_phase_fb = ""
            self._current_progress = 0.0

    # ═══════════════════════════════════════════════════════
    #  액션 클라이언트
    # ═══════════════════════════════════════════════════════

    def _send_pick_place(self, det, bin_id: str) -> bool:
        if self._action_client is None:
            self.get_logger().warn("[ACTION] No client (sim: success)")
            time.sleep(2.0)
            return True

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("[ACTION] Server not available.")
            return False

        goal          = PickPlace.Goal()
        goal.detection_id  = det.id
        goal.label         = det.label
        goal.pick_cx       = det.cx
        goal.pick_cy       = det.cy
        goal.has_3d        = det.has_3d
        goal.pick_x_m      = det.x_m
        goal.pick_y_m      = det.y_m
        goal.pick_z_m      = det.z_m
        goal.pick_angle_deg = float(getattr(det, "angle_deg", 0.0))
        goal.bin_id        = bin_id

        send_fut = self._action_client.send_goal_async(
            goal, feedback_callback=self._feedback_cb)

        if not self._wait_future(send_fut):
            return False

        gh = send_fut.result()
        if not gh.accepted:
            return False

        res_fut = gh.get_result_async()
        if not self._wait_future(res_fut):
            return False

        result = res_fut.result().result
        self.get_logger().info(
            f"[ACTION] success={result.success} msg='{result.message}'"
        )
        return result.success

    def _send_exec_command(self, cmd: str) -> bool:
        """SWEEP / HOME 특수 명령."""
        if self._action_client is None:
            self.get_logger().info(f"[EXEC] {cmd} (sim: success)")
            time.sleep(3.0)
            return True

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            return False

        goal          = PickPlace.Goal()
        goal.detection_id = -1
        goal.label        = cmd
        goal.bin_id       = cmd
        goal.has_3d       = False

        send_fut = self._action_client.send_goal_async(goal)
        if not self._wait_future(send_fut):
            return False

        gh = send_fut.result()
        if not gh.accepted:
            return False

        res_fut = gh.get_result_async()
        if not self._wait_future(res_fut):
            return False

        return res_fut.result().result.success

    def _wait_future(self, future, timeout: float = None) -> bool:
        timeout = timeout or self._action_timeout
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        return future.done()

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        with self._lock:
            self._current_phase_fb = fb.phase
            self._current_progress = fb.progress

    # ═══════════════════════════════════════════════════════
    #  상태 발행
    # ═══════════════════════════════════════════════════════

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
            s      = String()
            s.data = (
                f"state={self._state.value} phase={self._phase.value} "
                f"total={self._counters['total']} msg={self._last_message}"
            )
            self._status_pub.publish(s)


def main(args=None):
    rclpy.init(args=args)
    node     = ManagerNode()
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
