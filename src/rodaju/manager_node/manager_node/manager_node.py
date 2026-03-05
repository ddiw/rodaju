#!/usr/bin/env python3
"""
manager_node.py  ─  중앙제어 노드
═══════════════════════════════════════════════════════════════

[시나리오 흐름]
  PHASE 1. STANDBY        ─ 초기 대기
  PHASE 2. BAG_PICKUP     ─ 쓰레기 봉투 감지 → 집기 → 테이블에 붓기
  PHASE 3. SWEEP          ─ 주걱으로 쓰레기 훑어 이격
  PHASE 4. SORTING        ─ 음성/UI 명령으로 지정된 종류 순서대로 분류
  PHASE 5. PAUSED         ─ 일시정지 (현재 동작 완료 후 대기)
  PHASE 6. DONE           ─ 완료 후 홈 복귀

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

from std_msgs.msg import String

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
    STANDBY    = "STANDBY"
    BAG_PICKUP = "BAG_PICKUP"
    SWEEP      = "SWEEP"
    SORTING    = "SORTING"
    PAUSED     = "PAUSED"
    DONE       = "DONE"

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
    "water_bottle" : "BIN_PLASTIC",
    "can"          : "BIN_CAN",       # 캔
    "metal"        : "BIN_CAN",
    "aluminum"     : "BIN_CAN",
    "paper_cup"    : "BIN_PAPER",     # 종이컵
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
        self.declare_parameter("retry_limit",          2)

        self._status_rate    = self.get_parameter("status_rate").value
        self._action_timeout = self.get_parameter("action_timeout").value
        self._retry_limit    = self.get_parameter("retry_limit").value

        # ── 내부 상태 ────────────────────────────────────────
        self._lock         = threading.Lock()
        self._phase        = Phase.STANDBY
        self._state        = SystemState.IDLE
        self._priority     = "NONE"
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
        self._seen_ids: set[int] = set()

        # BAG_PICKUP 페이즈용 봉투 감지 결과 저장
        self._pending_bag_det = None   # Detection2D | None

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

        # vision_node 감지 모드 전환 (BAG / TRASH)
        # std_msgs/String 사용 – 인터페이스 설치 여부 무관
        self._vision_mode_pub = self.create_publisher(
            String, "/recycle/vision/mode", 10)

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
            f"[CMD] cmd={cmd} mode={mode} priority={msg.priority} "
            f"exclude={msg.exclude_mask:#04x} raw='{msg.raw_text}'"
        )

        with self._lock:
            if cmd == "START":
                if mode in ("sorting", ""):
                    self._phase = Phase.BAG_PICKUP
                    self._state = SystemState.RUNNING
                    self._last_message = "Starting scenario: bag pickup."
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
                if msg.priority:
                    self._priority = msg.priority
                self._exclude_mask = msg.exclude_mask
                # target 업데이트 (target 필드 없을 시 raw_text 에서 파싱 가능)
                self._last_message = (
                    f"Policy updated: priority={self._priority} "
                    f"exclude_mask={self._exclude_mask:#04x}"
                )

    # ═══════════════════════════════════════════════════════
    #  비전 콜백
    # ═══════════════════════════════════════════════════════

    def _vision_cb(self, msg):
        """
        페이즈별 처리:
          BAG_PICKUP → 봉투(trash_bag) 감지 결과를 _pending_bag_det 에 저장
          SORTING    → 쓰레기 감지 결과를 우선순위 큐에 삽입
        """
        if self._phase == Phase.BAG_PICKUP:
            # 봉투 레이블만 받아서 저장 (첫 번째 감지 결과 사용)
            for det in msg.detections:
                if det.label.lower() in ("trash_bag", "bag", "plastic_bag"):
                    if self._pending_bag_det is None:
                        self._pending_bag_det = det
                        self.get_logger().info(
                            f"[VISION] Bag detected: id={det.id} "
                            f"3d=({det.x_m:.3f},{det.y_m:.3f},{det.z_m:.3f})"
                        )
                    break
            return

        if self._phase != Phase.SORTING:
            return

        # SORTING 페이즈: 쓰레기 감지 큐 삽입
        for det in msg.detections:
            if det.id in self._seen_ids:
                continue

            bin_id = LABEL_TO_BIN.get(det.label.lower())
            if not bin_id:
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
        if self._priority == cat:
            return 0
        return PRIORITY_DEFAULT.get(cat, 99)

    # ═══════════════════════════════════════════════════════
    #  메인 워커 스레드 (시나리오 제어)
    # ═══════════════════════════════════════════════════════

    def _main_worker(self):
        self.get_logger().info("[WORKER] Main worker started.")

        while rclpy.ok():
            phase = self._phase

            if phase == Phase.STANDBY:
                time.sleep(0.3)

            elif phase == Phase.BAG_PICKUP:
                self._run_bag_pickup()

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
    #  PHASE 2: 봉투 집어서 테이블에 붓기
    # ─────────────────────────────────────────────────────────

    def _run_bag_pickup(self):
        self.get_logger().info("[PHASE] BAG_PICKUP")

        # vision_node 를 BAG 감지 모드로 전환
        self._publish_vision_mode("BAG")

        with self._lock:
            self._pending_bag_det  = None   # 이전 감지 결과 초기화
            self._current_label    = "trash_bag"
            self._current_bin      = "TABLE"
            self._current_phase_fb = "DETECTING_BAG"
            self._last_message     = "Detecting trash bag..."

        # 봉투 감지 대기 (최대 10초)
        self.get_logger().info("[BAG_PICKUP] Waiting for bag detection...")
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if self._pending_bag_det is not None:
                break
            time.sleep(0.1)

        bag_det = self._pending_bag_det
        if bag_det is None:
            self.get_logger().warn("[BAG_PICKUP] Bag not detected within timeout.")
        else:
            self.get_logger().info(
                f"[BAG_PICKUP] Bag at 3D=({bag_det.x_m:.3f},"
                f"{bag_det.y_m:.3f},{bag_det.z_m:.3f})m"
            )

        # exec_node 에 봉투 좌표 포함한 액션 전송
        success = self._send_bag_pickup(bag_det)

        with self._lock:
            self._current_label    = ""
            self._current_bin      = ""
            self._current_phase_fb = ""
            self._pending_bag_det  = None
            if success:
                self._last_message = "Bag emptied onto table. Starting sweep."
                self._phase        = Phase.SWEEP
            else:
                self._last_message = "Bag pickup failed. Back to standby."
                self._phase        = Phase.STANDBY

    # ─────────────────────────────────────────────────────────
    #  PHASE 3: 훑기
    # ─────────────────────────────────────────────────────────

    def _run_sweep(self):
        self.get_logger().info("[PHASE] SWEEP")
        with self._lock:
            self._current_phase_fb = "SWEEP"
            self._last_message     = "Sweeping trash to separate items..."

        success = self._send_exec_command("SWEEP")

        # SORTING 진입 전:
        #   1. vision_node 를 TRASH 감지 모드로 전환
        #   2. 이전 감지 ID / 큐 초기화 (새 봉투 내용물을 처음부터 인식)
        self._publish_vision_mode("TRASH")
        with self._lock:
            self._seen_ids.clear()
            # PriorityQueue 내부 큐 비우기
            while not self._det_queue.empty():
                try:
                    self._det_queue.get_nowait()
                except Exception:
                    break
            self._current_phase_fb = ""
            self._last_message = (
                "Sweep complete. Waiting for sort command."
                if success else
                "Sweep failed, proceeding to sort."
            )
            self._phase = Phase.SORTING

    # ─────────────────────────────────────────────────────────
    #  PHASE 4: 분류 (1 아이템 단위)
    # ─────────────────────────────────────────────────────────

    def _run_sorting_step(self):
        # PAUSE 체크
        if not self._pause_event.is_set():
            return

        try:
            prio, seq, det = self._det_queue.get(timeout=1.0)
        except queue.Empty:
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
        self._seen_ids.add(det.id)

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
                self._last_message = f"Failed: {det.label}. Will retry next cycle."
                self._seen_ids.discard(det.id)   # 실패 시 재시도 허용

            self._current_label    = ""
            self._current_bin      = ""
            self._current_phase_fb = ""
            self._current_progress = 0.0

    # ═══════════════════════════════════════════════════════
    #  액션 클라이언트
    # ═══════════════════════════════════════════════════════

    def _send_pick_place(self, det, bin_id: str, retry: int = 0) -> bool:
        if self._action_client is None:
            self.get_logger().warn("[ACTION] No client (sim: success)")
            time.sleep(2.0)
            return True

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("[ACTION] Server not available.")
            return False

        goal          = PickPlace.Goal()
        goal.detection_id = det.id
        goal.label        = det.label
        goal.pick_cx      = det.cx
        goal.pick_cy      = det.cy
        goal.has_3d       = det.has_3d
        goal.pick_x_m     = det.x_m
        goal.pick_y_m     = det.y_m
        goal.pick_z_m     = det.z_m
        goal.bin_id       = bin_id

        send_fut = self._action_client.send_goal_async(
            goal, feedback_callback=self._feedback_cb)

        if not self._wait_future(send_fut):
            return False

        gh = send_fut.result()
        if not gh.accepted:
            if retry < self._retry_limit:
                time.sleep(1.0)
                return self._send_pick_place(det, bin_id, retry + 1)
            return False

        res_fut = gh.get_result_async()
        if not self._wait_future(res_fut):
            if retry < self._retry_limit:
                return self._send_pick_place(det, bin_id, retry + 1)
            return False

        result = res_fut.result().result
        self.get_logger().info(
            f"[ACTION] success={result.success} msg='{result.message}'"
        )
        if not result.success and retry < self._retry_limit:
            time.sleep(0.5)
            return self._send_pick_place(det, bin_id, retry + 1)
        return result.success

    def _send_exec_command(self, cmd: str) -> bool:
        """BAG_PICKUP / SWEEP / HOME 특수 명령."""
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

    def _send_bag_pickup(self, bag_det) -> bool:
        """
        봉투 집기 액션 전송.
        bag_det 이 있으면 실제 3D 좌표를 goal 에 포함,
        없으면 (감지 실패) 좌표 없이 BAG_PICKUP 특수 명령만 전송.
        exec_node 는 label="BAG_PICKUP" 으로 분기 처리.
        """
        if self._action_client is None:
            self.get_logger().info("[BAG_PICKUP] sim: success")
            time.sleep(3.0)
            return True

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("[BAG_PICKUP] Action server not available.")
            return False

        goal              = PickPlace.Goal()
        goal.label        = "BAG_PICKUP"
        goal.bin_id       = "BAG_PICKUP"

        if bag_det is not None:
            # 감지된 봉투 좌표 전달
            goal.detection_id = bag_det.id
            goal.pick_cx      = bag_det.cx
            goal.pick_cy      = bag_det.cy
            goal.has_3d       = bag_det.has_3d
            goal.pick_x_m     = bag_det.x_m
            goal.pick_y_m     = bag_det.y_m
            goal.pick_z_m     = bag_det.z_m
        else:
            # 감지 실패 → 좌표 없이 전달 (exec_node 에서 현재 포즈 기준으로 처리)
            goal.detection_id = -1
            goal.has_3d       = False
            goal.pick_x_m     = 0.0
            goal.pick_y_m     = 0.0
            goal.pick_z_m     = 0.0
            self.get_logger().warn("[BAG_PICKUP] No bag detected – sending without coords.")

        send_fut = self._action_client.send_goal_async(
            goal, feedback_callback=self._feedback_cb)
        if not self._wait_future(send_fut):
            return False

        gh = send_fut.result()
        if not gh.accepted:
            self.get_logger().warn("[BAG_PICKUP] Goal rejected.")
            return False

        res_fut = gh.get_result_async()
        if not self._wait_future(res_fut):
            self.get_logger().error("[BAG_PICKUP] Result timeout.")
            return False

        result = res_fut.result().result
        self.get_logger().info(
            f"[BAG_PICKUP] success={result.success} msg='{result.message}'"
        )
        return result.success

    def _publish_vision_mode(self, mode: str):
        """
        vision_node 에 감지 모드 변경 요청.
          mode = "BAG"   → 봉투 감지 모드
          mode = "TRASH" → 쓰레기 분류 감지 모드
        """
        msg      = String()
        msg.data = mode
        self._vision_mode_pub.publish(msg)
        self.get_logger().info(f"[VISION MODE] → {mode}")

    # ═══════════════════════════════════════════════════════
    #  상태 발행
    # ═══════════════════════════════════════════════════════

    def _publish_status(self):
        try:
            msg = SystemStatus()
            msg.stamp             = self.get_clock().now().to_msg()
            msg.state             = self._state.value
            msg.mode              = self._phase.value
            msg.priority          = self._priority
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
