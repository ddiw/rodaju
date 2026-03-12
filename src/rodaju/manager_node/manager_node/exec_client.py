#!/usr/bin/env python3
"""PickPlace 액션 클라이언트 래퍼."""

import time

from rclpy.action import ActionClient

try:
    from recycle_interfaces.action import PickPlace
    INTERFACES_AVAILABLE = True
except ImportError:
    INTERFACES_AVAILABLE = False


class ExecActionClient:
    """exec_node의 PickPlace 액션 서버와 통신하는 클라이언트.

    Args:
        node:        rclpy Node (ActionClient 생성 및 로거 사용)
        timeout:     액션 응답 대기 시간 (초)
        feedback_cb: feedback 수신 시 호출되는 콜백 (phase: str, progress: float) → None
        cb_group:    ActionClient 콜백 그룹 (optional)
    """

    def __init__(self, node, timeout: float = 40.0, feedback_cb=None, cb_group=None):
        self._logger = node.get_logger()
        self._timeout = timeout
        self._feedback_cb = feedback_cb

        if INTERFACES_AVAILABLE:
            self._client = ActionClient(
                node, PickPlace, "/recycle/exec/pick_place",
                callback_group=cb_group,
            )
        else:
            self._client = None

    # ── 공개 API ──────────────────────────────────────────────

    def send_pick_place(self, det, bin_id: str) -> bool:
        """Detection 결과를 exec_node로 전송해 pick & place 수행."""
        if self._client is None:
            self._logger.warn("[ACTION] No client (sim: success)")
            time.sleep(2.0)
            return True

        if not self._client.wait_for_server(timeout_sec=5.0):
            self._logger.error("[ACTION] Server not available.")
            return False

        goal               = PickPlace.Goal()
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

        send_fut = self._client.send_goal_async(goal, feedback_callback=self._on_feedback)
        if not self._wait(send_fut):
            return False

        gh = send_fut.result()
        if not gh.accepted:
            return False

        res_fut = gh.get_result_async()
        if not self._wait(res_fut):
            return False

        result = res_fut.result().result
        self._logger.info(f"[ACTION] success={result.success} msg='{result.message}'")
        return result.success

    def send_exec_command(self, cmd: str) -> bool:
        """SWEEP / HOME / GOTO_WORK / CLEAN_DESK 등 특수 명령 전송."""
        if self._client is None:
            self._logger.info(f"[EXEC] {cmd} (sim: success)")
            time.sleep(3.0)
            return True

        if not self._client.wait_for_server(timeout_sec=5.0):
            return False

        goal              = PickPlace.Goal()
        goal.detection_id = -1
        goal.label        = cmd
        goal.bin_id       = cmd
        goal.has_3d       = False

        send_fut = self._client.send_goal_async(goal)
        if not self._wait(send_fut):
            return False

        gh = send_fut.result()
        if not gh.accepted:
            return False

        res_fut = gh.get_result_async()
        if not self._wait(res_fut):
            return False

        return res_fut.result().result.success

    # ── 내부 헬퍼 ─────────────────────────────────────────────

    def _wait(self, future) -> bool:
        deadline = time.time() + self._timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        return future.done()

    def _on_feedback(self, feedback_msg):
        if self._feedback_cb:
            fb = feedback_msg.feedback
            self._feedback_cb(fb.phase, fb.progress)
