#!/usr/bin/env python3
"""
m0609_exec_node.py  ─  로봇 동작 수행 노드
═══════════════════════════════════════════════════════════════

[담당 동작]
  1. BAG_PICKUP  ─ 고정 좌표로 봉투 집기 → 테이블 위에서 뒤집어 붓기
  2. SWEEP       ─ 주걱 동작으로 쓰레기 이격
  3. PICK_PLACE  ─ 쓰레기 집기 → 종류별 분류함에 투입
  4. HOME        ─ 홈 포지션 복귀

[특징]
  - 종류별 그리퍼 힘/폭 제어
  - 파지 실패 시 재시도
"""

import os
import time

from ament_index_python.packages import get_package_share_directory

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
import DR_init

# ── Hand-Eye 캘리브레이션 파일 경로 ──────────────────────────────
T_GRIPPER2CAM_PATH = os.path.join(
    get_package_share_directory("execute_node"), "resource", "T_gripper2camera.npy"
)

from recycle_interfaces.action import PickPlace

# ── DSR 로봇 API ──────────────────────────────────────────────
ROBOT_ID    = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY    = 100
ACC         = 60
VELOCITY_SLOW = 50    # 훑기 / 정밀 동작용

DR_init.__dsr__id    = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
_dsr_node = rclpy.create_node("m0609_exec_dsr_init", namespace=ROBOT_ID)
DR_init.__dsr__node = _dsr_node

from DSR_ROBOT2 import movej, movel, get_current_posx, mwait, trans
from execute_node.onrobot import RG


# ═══════════════════════════════════════════════════════════════
#  위치 / 파라미터 상수
# ═══════════════════════════════════════════════════════════════

# 홈 조인트 각도
J_HOME = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]

# 테이블 위 작업 시작 관절 자세 (J_HOME 이후 movel singularity 방지용)
# ※ 로봇을 직접 조그해서 테이블 위 안전 위치로 이동 후 관절값을 읽어 입력
J_WORK = [-11.0, 26.0, 19.0, 0.0, 133.0, -12.0]
# 테이블 중심
TABLE_CENTER_POS = [500.0, 0.0, 90.0, 180.0, 0.0, 90.0]


# 모으기 동작 파라미터
GATHER_LIFT_Z = 200.0   # 이동 시 들어올리는 높이 오프셋 (mm)

# (시작 위치, 끝 위치) 쌍 – z=133.8이 접촉 높이
GATHER_STEPS = [
    ([838.8, -180.0, 133.8,   0.0, 150.0,   0.0], [638.8, -180.0, 133.8,   0.0, 150.0,   0.0]),
    ([838.8,    0.0, 133.8,   0.0, 150.0,   0.0], [638.8,    0.0, 133.8,   0.0, 150.0,   0.0]),
    ([838.8,  170.0, 133.8,   0.0, 150.0,   0.0], [638.8,  170.0, 133.8,   0.0, 150.0,   0.0]),
    ([331.6, -180.0, 133.8, 180.0, 160.0, 180.0], [431.6, -180.0, 133.8, 180.0, 160.0, 180.0]),
    ([331.6,    0.0, 133.8, 180.0, 160.0, 180.0], [431.6,    0.0, 133.8, 180.0, 160.0, 180.0]),
    ([331.6,  170.0, 133.8, 180.0, 160.0, 180.0], [431.6,  170.0, 133.8, 180.0, 160.0, 180.0]),
]

# 빗자루 거치대 위치 (로봇 베이스 좌표계) – 실제 값으로 교체 필요
BROOM_HOLDER_POS    = [416.0, 285.0, 160.0, 180.0, 180.0, 90.0]
BROOM_APPROACH_Z    = 150.0   # 거치대 위 접근 높이 오프셋 (mm)

# pick & place
APPROACH_Z_OFFSET = 80.0
PICK_Z_OFFSET     = -30.0
MAX_GRASP_RETRIES = 3

# 분류함 위치 (로봇 베이스 좌표계)
BIN_POSITIONS: dict[str, list] = {
    "BIN_PLASTIC": [446.0, -520.0, 260.0, 90.0, -150.0, 90.0],   # 500ml 페트병
    "BIN_CAN"    : [280.0, -520.0, 260.0, 90.0, -150.0, 90.0],   # 캔
    "BIN_PAPER"  : [120.0, -520.0, 260.0, 90.0, -150.0, 90.0],   # 종이컵
}

# 분류 종류별 그리퍼 파라미터 (force: 1/10 N, width: 1/10 mm)
GRIPPER_PARAMS: dict[str, dict] = {
    "BIN_PLASTIC": {"force": 300, "width": 1000},   # 페트병 (부드럽게)
    "BIN_CAN"    : {"force": 350, "width": 1000},   # 캔 (단단하게)
    "BIN_PAPER"  : {"force": 200, "width": 1000},   # 종이컵 (매우 부드럽게)
    "DEFAULT"    : {"force": 200, "width": 1000},
}


# ═══════════════════════════════════════════════════════════════
#  로봇 / 그리퍼 래퍼
# ═══════════════════════════════════════════════════════════════

class RobotAPI:
    def movej(self, joints, vel=VELOCITY, acc=ACC):
        movej(joints, vel=vel, acc=acc)

    def movel(self, pos, vel=VELOCITY, acc=ACC):
        movel(pos, vel=vel, acc=acc)

    def mwait(self):
        mwait()

    def get_posx(self) -> list:
        return list(get_current_posx()[0])

    def trans_offset(self, pos: list, delta: list) -> list:
        return list(trans(pos, delta))


class GripperAPI:
    def __init__(self):
        self._g = RG("rg2", "192.168.1.1", 502)

    def open(self, force=400):
        self._g.open_gripper(force_val=force)
        self._wait()

    def close(self, force=400):
        self._g.close_gripper(force_val=force)
        self._wait()

    def move(self, width: int, force: int = 200):
        self._g.move_gripper(width_val=width, force_val=force)
        self._wait()

    def is_gripping(self) -> bool:
        return bool(self._g.get_status()[1])

    def _wait(self, timeout=5.0):
        t = time.time()
        while time.time() - t < timeout:
            if not self._g.get_status()[0]:
                break
            time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════
#  M0609ExecNode
# ═══════════════════════════════════════════════════════════════

class M0609ExecNode(Node):

    def __init__(self):
        super().__init__("m0609_exec_node")

        # OBB 각도→그리퍼 RZ 보정값 (카메라 마운팅에 따라 조정)
        # 예: 카메라 x축이 로봇 y축 방향이면 90.0, 반대면 -90.0
        self.declare_parameter("gripper_angle_offset_deg", -90.0)
        self._gripper_angle_offset = self.get_parameter("gripper_angle_offset_deg").value

        self.robot   = RobotAPI()
        self.gripper = GripperAPI()
        self._busy   = False

        try:
            self._T_gripper2cam = np.load(T_GRIPPER2CAM_PATH)
            self.get_logger().info(
                f"Hand-Eye matrix loaded: {T_GRIPPER2CAM_PATH}\n"
                f"{self._T_gripper2cam}"
            )
        except Exception as e:
            self._T_gripper2cam = None
            self.get_logger().warn(
                f"T_gripper2camera.npy load failed: {e}\n"
                "-> falling back to pixel estimate."
            )

        # ── 액션 서버 ──────────────────────────────────────
        self._action_server = ActionServer(
            self,
            PickPlace,
            "/recycle/exec/pick_place",
            execute_callback   = self._execute_cb,
            goal_callback      = self._goal_cb,
            cancel_callback    = self._cancel_cb,
            callback_group     = ReentrantCallbackGroup(),
        )
        self.get_logger().info("PickPlace action server ready.")

        self._go_home()
        self.get_logger().info("M0609ExecNode ready.")

    # ══════════════════════════════════════════════════════
    #  액션 서버 콜백
    # ══════════════════════════════════════════════════════

    def _goal_cb(self, goal_request):
        if self._busy:
            self.get_logger().warn("Goal rejected – robot busy.")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, _):
        self.get_logger().info("Cancel requested.")
        return CancelResponse.ACCEPT

    def _execute_cb(self, goal_handle):
        self._busy = True
        goal     = goal_handle.request
        feedback = PickPlace.Feedback()
        result   = PickPlace.Result()

        def fb(phase: str, progress: float):
            feedback.phase    = phase
            feedback.progress = progress
            goal_handle.publish_feedback(feedback)
            self.get_logger().info(f"  [FB] {phase} {progress:.0f}%")

        try:
            # ── 특수 명령 분기 ──────────────────────────────
            if goal.label in ("SWEEP", "HOME", "GOTO_WORK"):
                if goal.label == "SWEEP":
                    success = self._do_sweep(fb)
                elif goal.label == "GOTO_WORK":
                    self.robot.movej(J_WORK, vel=VELOCITY)
                    self.robot.mwait()
                    success = True
                else:
                    self._go_home(); success = True

            # ── 일반 pick & place ───────────────────────────
            else:
                success = self._do_pick_place(goal, fb)

            result.success = success
            result.message = "OK" if success else "FAILED"
            if success:
                goal_handle.succeed()
            else:
                goal_handle.abort()

        except Exception as e:
            self.get_logger().error(f"Execute error: {e}")
            result.success = False
            result.message = str(e)
            fb("ERROR", 0.0)
            self._go_home()
            goal_handle.abort()
        finally:
            self._busy = False

        return result

    # ══════════════════════════════════════════════════════
    #  동작 1: 주걱 훑기
    # ══════════════════════════════════════════════════════

    def _do_sweep(self, fb) -> bool:
        """
        1. 거치대에서 빗자루 파지
        2. GATHER_STEPS 순서대로 쓰레기 모으기
        3. 거치대에 빗자루 반납 → 홈
        """
        fb("SWEEP_START", 5.0)
        self.get_logger().info("[SWEEP] Starting gather sequence.")

        broom_above = list(BROOM_HOLDER_POS)
        broom_above[2] += BROOM_APPROACH_Z

        # 1. 거치대 접근 → 파지
        fb("BROOM_PICK", 10.0)
        self.gripper.open()
        self.robot.movel(broom_above, vel=VELOCITY_SLOW, acc=ACC)
        self.robot.mwait()
        self.robot.movel(BROOM_HOLDER_POS, vel=VELOCITY_SLOW, acc=ACC)
        self.robot.mwait()
        self.gripper.close(force=GRIPPER_PARAMS["DEFAULT"]["force"])
        self.robot.movel(broom_above, vel=VELOCITY_SLOW, acc=ACC)
        self.robot.mwait()
        self.robot.movej(J_WORK, vel=VELOCITY_SLOW, acc=ACC)
        self.robot.mwait()

        # 2. 모으기 동작
        n = len(GATHER_STEPS)
        for i, (start, end) in enumerate(GATHER_STEPS):
            start_above = list(start)
            start_above[2] += GATHER_LIFT_Z
            end_above = list(end)
            end_above[2] += GATHER_LIFT_Z

            # 시작 위치 위로 이동 → 내려서 접촉 → 밀기 → 들어올리기
            self.robot.movel(start_above, vel=VELOCITY, acc=ACC)
            self.robot.mwait()
            self.robot.movel(start, vel=VELOCITY, acc=ACC)
            self.robot.mwait()
            self.robot.movel(end, vel=VELOCITY, acc=ACC)
            self.robot.mwait()
            self.robot.movel(end_above, vel=VELOCITY, acc=ACC)
            self.robot.mwait()

            progress = 15.0 + 70.0 * ((i + 1) / n)
            fb("GATHERING", progress)

        # 3. 거치대에 빗자루 반납
        fb("BROOM_RETURN", 90.0)
        self.robot.movej(J_WORK, vel=VELOCITY, acc=ACC)
        self.robot.mwait()
        self.robot.movel(broom_above, vel=VELOCITY_SLOW, acc=ACC)
        self.robot.mwait()
        self.robot.movel(BROOM_HOLDER_POS, vel=VELOCITY_SLOW, acc=ACC)
        self.robot.mwait()
        self.gripper.open()
        self.robot.movel(broom_above, vel=VELOCITY, acc=ACC)
        self.robot.mwait()

        self._go_home()
        fb("DONE", 100.0)
        self.get_logger().info("[SWEEP] Gather complete.")
        return True

    # ══════════════════════════════════════════════════════
    #  동작 3: Pick & Place
    # ══════════════════════════════════════════════════════

    def _do_pick_place(self, goal, fb) -> bool:
        """
        1. 목표 좌표 산출 (3D 우선, 없으면 픽셀 추정)
        2. 접근 → 파지 → 들기 → 분류함 이동 → 투입 → 홈
        """
        bin_id    = goal.bin_id
        gp        = GRIPPER_PARAMS.get(bin_id, GRIPPER_PARAMS["DEFAULT"])
        bin_pos   = BIN_POSITIONS.get(bin_id, BIN_POSITIONS["BIN_PLASTIC"])

        # 목표 좌표
        if goal.has_3d:
            target = self._cam_to_robot(goal.pick_x_m, goal.pick_y_m, goal.pick_z_m)
        else:
            target = self._pixel_estimate(goal.pick_cx, goal.pick_cy)

        # OBB 각도로 그리퍼 RZ 설정
        # pick_angle_deg: 카메라 이미지 평면에서 객체 장축 방향 (°)
        # gripper_angle_offset: 카메라 마운팅 보정값 (파라미터로 조정)
        pick_rz = target[5] + goal.pick_angle_deg + self._gripper_angle_offset
        target[5] = pick_rz % 360.0
        self.get_logger().info(
            f"[PICK] label={goal.label} obb_angle={goal.pick_angle_deg:.1f}° "
            f"offset={self._gripper_angle_offset:.1f}° -> rz={target[5]:.1f}°"
        )
        self.get_logger().info(
            f"[PICK] cam=({goal.pick_x_m*1000:.1f},{goal.pick_y_m*1000:.1f},{goal.pick_z_m*1000:.1f})mm "
            f"-> target=({target[0]:.1f},{target[1]:.1f},{target[2]:.1f})mm "
            f"approach_z={target[2]+APPROACH_Z_OFFSET:.1f}mm pick_z={target[2]+PICK_Z_OFFSET:.1f}mm"
        )

        approach  = list(target); approach[2]  += APPROACH_Z_OFFSET
        pick_pos  = list(target); pick_pos[2]  += PICK_Z_OFFSET
        bin_above = list(bin_pos); bin_above[2] += APPROACH_Z_OFFSET

        self._transit_to_work()

        # 1. APPROACH
        fb("APPROACH", 10.0)
        self.gripper.move(width=gp["width"], force=gp["force"])
        self.robot.movel(approach)
        self.robot.mwait()

        # 2. GRASP (파지 실패 시 재시도)
        gripped = False
        for attempt in range(1, MAX_GRASP_RETRIES + 1):
            fb("GRASP", 30.0)
            self.robot.movel(pick_pos, vel=VELOCITY_SLOW)
            self.robot.mwait()
            self.gripper.close(force=gp["force"])

            if self.gripper.is_gripping():
                gripped = True
                break

            self.get_logger().warn(
                f"[GRASP] attempt {attempt}/{MAX_GRASP_RETRIES} – no object detected, retrying."
            )
            # 다시 접근 위치로 올라가서 재시도
            self.robot.movel(approach, vel=VELOCITY_SLOW)
            self.robot.mwait()
            self.gripper.move(width=gp["width"], force=gp["force"])

        if not gripped:
            self.get_logger().error("[GRASP] Failed to grasp after all retries.")
            self.robot.movel(approach)
            self.robot.mwait()
            return False

        # 3. LIFT
        fb("LIFT", 50.0)
        self.robot.movel(approach)
        self.robot.mwait()

        fb("MOVE_HOME", 60.0)
        self.robot.movej(J_HOME)
        self.robot.mwait()

        # 4. MOVE_BIN (LIFT 위치에서 bin_above로 직접 이동 – J_HOME 경유 시 singularity 발생)
        fb("MOVE_BIN", 65.0)
        self.robot.movel(bin_above)
        self.robot.mwait()

        # 5. PLACE
        fb("PLACE", 85.0)
        self.robot.movel(bin_pos, vel=VELOCITY_SLOW)
        self.robot.mwait()
        self.gripper.open()
        time.sleep(0.3)

        # 6. RETURN
        fb("RETURN", 95.0)
        self.robot.movel(bin_above)
        self.robot.mwait()
        self._go_home()

        fb("DONE", 100.0)
        self.get_logger().info(f"[PICK_PLACE] {goal.label} → {bin_id} complete.")
        return True

    # ══════════════════════════════════════════════════════
    #  좌표 변환
    # ══════════════════════════════════════════════════════

    def _posx_to_matrix(self, posx: list) -> np.ndarray:
        """
        DSR get_current_posx() 결과 [x, y, z, rx, ry, rz] (mm, deg)
        -> 4x4 동차 변환행렬 T_base2gripper
        """
        x, y, z, rx, ry, rz = posx
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = [x, y, z]
        return T

    def _cam_to_robot(self, x_m: float, y_m: float, z_m: float) -> list:
        """
        카메라 좌표계(m) -> 로봇 베이스 좌표계(mm) 변환.
        """
        if self._T_gripper2cam is None:
            self.get_logger().warn("[CAM2ROBOT] No calibration matrix – using pixel estimate.")
            return self._pixel_estimate(320, 240)

        # 카메라 좌표 m -> mm
        coord = np.append(
            np.array([x_m * 1000.0, y_m * 1000.0, z_m * 1000.0]),
            1.0
        )

        # 현재 TCP 포즈 -> T_base2gripper
        robot_posx    = self.robot.get_posx()
        base2gripper  = self._posx_to_matrix(robot_posx)

        # 변환
        td_coord = np.dot(base2gripper @ self._T_gripper2cam, coord)  # shape (4,)

        # 후처리
        DEPTH_OFFSET = -5.0   # mm
        MIN_Z        =  2.0   # mm

        if td_coord[2] and np.sum(td_coord[:3]) != 0:
            td_coord[2] += DEPTH_OFFSET
            td_coord[2]  = max(td_coord[2], MIN_Z)

        # target_pos = td_coord[:3] + 현재 자세각
        target_pos = list(td_coord[:3]) + list(robot_posx[3:])

        self.get_logger().debug(
            f"[CAM2ROBOT] cam=({x_m*1000:.1f},{y_m*1000:.1f},{z_m*1000:.1f})mm "
            f"-> base=({td_coord[0]:.1f},{td_coord[1]:.1f},{td_coord[2]:.1f})mm"
        )
        return target_pos

    def _pixel_estimate(self, cx: int, cy: int) -> list:
        """Depth 없을 때 픽셀 기반 fallback."""
        current = self.robot.get_posx()
        SCALE = 0.5
        pos = list(current)
        pos[0] += (cx - 320) * SCALE
        pos[1] += (cy - 240) * SCALE
        pos[2]  = TABLE_CENTER_POS[2]
        self.get_logger().warn(f"[ESTIMATE] pixel fallback cx={cx} cy={cy}")
        return pos

    # ══════════════════════════════════════════════════════
    #  공통
    # ══════════════════════════════════════════════════════

    def _go_home(self):
        self.robot.movej(J_HOME, vel=VELOCITY, acc=ACC)
        self.robot.mwait()

    def _transit_to_work(self):
        """J_HOME 이후 작업 영역 진입 – movej로 singularity 없이 이동."""
        self.robot.movej(J_WORK)
        self.robot.mwait()


def main(args=None):
    node     = M0609ExecNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
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