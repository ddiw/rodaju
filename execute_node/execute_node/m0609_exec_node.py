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
  - DSR / OnRobot 미설치 환경에서 시뮬레이션 모드 자동 전환
"""

import os
import time
import threading

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

try:
    from recycle_interfaces.action import PickPlace,Detectinginit
    INTERFACES_AVAILABLE = True
except ImportError:
    INTERFACES_AVAILABLE = False

# ── DSR 로봇 API ──────────────────────────────────────────────
ROBOT_ID    = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY    = 60
ACC         = 60
VELOCITY_SLOW = 30    # 훑기 / 정밀 동작용

DR_init.__dsr__id    = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
_dsr_node = rclpy.create_node("m0609_exec_dsr_init", namespace=ROBOT_ID)
DR_init.__dsr__node = _dsr_node

try:
    from DSR_ROBOT2 import movej, movel, get_current_posx, mwait, trans, posx,posj
    DSR_AVAILABLE = True
except ImportError:
    DSR_AVAILABLE = False
    print("[WARN] DSR_ROBOT2 not available – simulation mode.")

try:
    from execute_node.onrobot import RG
    GRIPPER_AVAILABLE = True
except ImportError:
    GRIPPER_AVAILABLE = False
    print("[WARN] OnRobot not available – simulation mode.")


# ═══════════════════════════════════════════════════════════════
#  위치 / 파라미터 상수
# ═══════════════════════════════════════════════════════════════

# 홈 조인트 각도
J_HOME = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]

# 테이블 위 작업 시작 관절 자세 (J_HOME 이후 movel singularity 방지용)
# ※ 로봇을 직접 조그해서 테이블 위 안전 위치로 이동 후 관절값을 읽어 입력
J_WORK = [-11.0, 26.0, 19.0, 0.0, 133.0, -12.0]   # TODO: 실제 값으로 교체 필요

# 테이블 중심 (봉투를 붓는 위치)
TABLE_CENTER_POS = [350.0, 0.0, 120.0, 180.0, 0.0, 90.0]

# 봉투 집는 위치 위 접근 높이
BAG_APPROACH_Z_OFFSET = 100.0
BAG_GRIP_Z_OFFSET     = -15.0
# 붓는 동작: 테이블 위에서 뒤집기 (RZ 180도 회전)
POUR_ROTATION_OFFSET  = [0.0, 0.0, 150.0, 0.0, 0.0, 180.0]

# 훑기 파라미터
SWEEP_Z        = 85.0     # 테이블 표면 높이 (mm)
SWEEP_RANGE_X  = 250.0    # 좌우 훑기 범위 (mm)
SWEEP_RANGE_Y  = 180.0    # 앞뒤 훑기 범위 (mm)
SWEEP_STEP_Y   = 60.0     # Y 스텝 간격 (mm)

# pick & place
APPROACH_Z_OFFSET = 80.0
PICK_Z_OFFSET     = 0.0

# J_HOME에서 movel로 도달 가능한 안전 경유점 (singularity 방지용)
# PICK_PLACE 로그에서 이 위치는 J_HOME → movel로 정상 도달 확인됨
WORK_APPROACH_POS = [
    TABLE_CENTER_POS[0],
    TABLE_CENTER_POS[1],
    TABLE_CENTER_POS[2] + APPROACH_Z_OFFSET,
    180.0, 0.0, 90.0,
]

# 분류함 위치 (로봇 베이스 좌표계)
BIN_POSITIONS: dict[str, list] = {
    "BIN_PLASTIC": [150.0, -470.0, 400.0, 0.0, -180.0, 0.0],   # 500ml 페트병
    "BIN_CAN"    : [350.0, -470.0, 400.0, 0.0, -180.0, 0.0],   # 캔
    "BIN_PAPER"  : [550.0, -470.0, 400.0, 0.0, -180.0, 0.0],   # 종이컵
}

# 분류 종류별 그리퍼 파라미터 (force: 1/10 N, width: 1/10 mm)
GRIPPER_PARAMS: dict[str, dict] = {
    "BIN_PLASTIC": {"force": 150, "width": 1000},   # 페트병 (부드럽게)
    "BIN_CAN"    : {"force": 350, "width": 1000},   # 캔 (단단하게)
    "BIN_PAPER"  : {"force":  80, "width": 1000},   # 종이컵 (매우 부드럽게)
    "DEFAULT"    : {"force": 200, "width": 1000},
    "BAG"        : {"force": 400, "width": 1000},   # 봉투 (최대힘)
}


# ═══════════════════════════════════════════════════════════════
#  로봇 / 그리퍼 래퍼 (미설치 시 시뮬레이션)
# ═══════════════════════════════════════════════════════════════

class RobotAPI:
    def movej(self, joints, vel=VELOCITY, acc=ACC):
        if DSR_AVAILABLE:
            movej(joints, vel=vel, acc=acc)
        else:
            print(f"  [SIM] movej {joints}")

    def movel(self, pos, vel=VELOCITY, acc=ACC):
        if DSR_AVAILABLE:
            movel(pos, vel=vel, acc=acc)
        else:
            print(f"  [SIM] movel {[round(v,1) for v in pos]}")

    def mwait(self):
        if DSR_AVAILABLE:
            mwait()
        else:
            time.sleep(0.2)

    def get_posx(self) -> list:
        if DSR_AVAILABLE:
            return list(get_current_posx()[0])
        return [0.0, 0.0, 300.0, 180.0, 0.0, 90.0]

    def trans_offset(self, pos: list, delta: list) -> list:
        if DSR_AVAILABLE:
            return list(trans(pos, delta))
        result = list(pos)
        for i in range(min(len(delta), len(result))):
            result[i] += delta[i]
        return result


class GripperAPI:
    def __init__(self):
        self._g = None
        if GRIPPER_AVAILABLE:
            try:
                self._g = RG("rg2", "192.168.1.1", 502)
            except Exception as e:
                print(f"[WARN] Gripper init failed: {e}")

    def open(self, force=400):
        if self._g:
            self._g.open_gripper(force_val=force)
            self._wait()
        else:
            print("  [SIM] gripper open")

    def close(self, force=400):
        if self._g:
            self._g.close_gripper(force_val=force)
            self._wait()
        else:
            print("  [SIM] gripper close")

    def move(self, width: int, force: int = 200):
        if self._g:
            self._g.move_gripper(width_val=width, force_val=force)
            self._wait()
        else:
            print(f"  [SIM] gripper move w={width} f={force}")

    def is_gripping(self) -> bool:
        if self._g:
            return bool(self._g.get_status()[1])
        return True

    def _wait(self, timeout=5.0):
        if not self._g:
            return
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
        self._lock   = threading.Lock()

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
        if INTERFACES_AVAILABLE:
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
        else:
            self.get_logger().warn("recycle_interfaces not found – action server disabled.")

        self._transit_to_work()
        self.get_logger().info("M0609ExecNode ready.")

        self.det_init_action_server = ActionServer(
                self,
                Detectinginit,
                "/recycle/exec/detect_pos",
                execute_callback   = self._execute_cb,
                goal_callback      = self._goal_cb,
                cancel_callback    = self._cancel_cb,
                callback_group     = ReentrantCallbackGroup(),
            )
        self.get_logger().info(" Detectinginit server ready.")

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
            if goal.label in ("BAG_PICKUP", "SWEEP", "HOME"):
                if goal.label == "BAG_PICKUP":
                    success = self._do_bag_pickup(goal, fb)
                elif goal.label == "SWEEP":
                    success = self._do_sweep(fb)
                else:
                    self._go_home(); success = True

            elif goal.label == "detect":  succes = self.detecting_position(goal,fb)
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

    # 동작 0: 디텍팅 위치로 이동 
    def detecting_position(self, fb)->bool:
        self.get_logger().info("detecting position move!")
        self.robot.movej(J_WORK)
        self.robot.mwait()

    # ══════════════════════════════════════════════════════
    #  동작 1: 봉투 집어서 테이블에 붓기
    # ══════════════════════════════════════════════════════

    def _do_bag_pickup(self, goal, fb) -> bool:
        """
        1. 고정 좌표로 봉투 위치 산출
        2. 봉투 위 접근
        3. 봉투 파지 (최대 힘)
        4. 들어올리기
        5. 테이블 중앙으로 이동
        6. 뒤집어 붓기 (Z축 회전)
        7. 그리퍼 열기 → 홈
        """
        self.get_logger().info("[BAG_PICKUP] Starting bag pickup sequence.")
        # self._transit_to_work()

        # J_WORK 자세각 사용 – orientation mismatch 방지
        ori = self.robot.get_posx()[3:]

        gp = GRIPPER_PARAMS["BAG"]

        # manager_node 에서 전달받은 고정 좌표 사용
        bag_pos = self._cam_to_robot(goal.pick_x_m, goal.pick_y_m, goal.pick_z_m)
        self.get_logger().info(
            f"[BAG_PICKUP] Target: ({bag_pos[0]:.1f},{bag_pos[1]:.1f},{bag_pos[2]:.1f})mm"
        )

        # 1. 접근
        fb("APPROACH", 10.0)
        approach = list(bag_pos)
        approach[2] += BAG_APPROACH_Z_OFFSET
        self.robot.movel(approach)
        self.robot.mwait()

        # 2. 파지
        fb("GRASP_BAG", 30.0)
        self.gripper.move(width=gp["width"], force=gp["force"])
        self.robot.movel(bag_pos)
        self.robot.mwait()
        self.gripper.close(force=gp["force"])

        if not self.gripper.is_gripping():
            self.get_logger().warn("[BAG_PICKUP] Grip failed – retrying.")
            self.gripper.close(force=400)

        # 3. 들어올리기
        fb("LIFT", 45.0)
        lift_pos = list(bag_pos)
        lift_pos[2] += BAG_APPROACH_Z_OFFSET 
        self.robot.movel(lift_pos)
        self.robot.mwait()

        # 4. 테이블 중앙으로 이동
        fb("MOVE_TO_TABLE", 60.0)
        pour_approach = TABLE_CENTER_POS[:3] + [ori[0], ori[1], (ori[2] + 0.0) % 360.0]
        pour_approach[2] += 100.0
        self.robot.movel(pour_approach)
        self.robot.mwait()

        # 5. 뒤집어 붓기 (TCP 회전으로 봉투 역전)
        fb("POUR", 75.0)
        pour_pos = TABLE_CENTER_POS[:3] + [ori[0], ori[1], (ori[2] + 180.0) % 360.0]
        self.robot.movel(pour_pos, vel=VELOCITY_SLOW)
        self.robot.mwait()
        self.gripper.open(force=gp["force"])
        time.sleep(1.0)        # 내용물 낙하 대기

        # 6. 복귀
        fb("RETURN", 90.0)
        self.robot.movel(pour_approach)
        self.robot.mwait()
        self._go_home()

        fb("DONE", 100.0)
        self.get_logger().info("[BAG_PICKUP] Complete.")
        return True

    # ══════════════════════════════════════════════════════
    #  동작 2: 주걱 훑기 (쓰레기 이격)
    # ══════════════════════════════════════════════════════

    def _do_sweep(self, fb) -> bool:
        """
        테이블 위 쓰레기들을 주걱 엔드이펙터로 좌우 빗질하여 간격 확보.
        Y축 방향으로 스텝 이동하며 X 방향 왕복.
        """
        fb("SWEEP_START", 5.0)
        self.get_logger().info("[SWEEP] Starting sweep sequence.")
        # self._transit_to_work()

        # J_WORK 자세각을 그대로 사용 – 하드코딩 orientation으로 movel 하면
        # arm configuration mismatch로 DSR이 거부함
        ori = self.robot.get_posx()[3:]

        cx, cy = TABLE_CENTER_POS[0], TABLE_CENTER_POS[1]
        n_steps = int(SWEEP_RANGE_Y / SWEEP_STEP_Y) + 1

        for i, step in enumerate(range(n_steps)):
            y_pos = cy - SWEEP_RANGE_Y / 2 + step * SWEEP_STEP_Y

            # 해당 Y 줄 접근 (위에서)
            above = [cx - SWEEP_RANGE_X / 2, y_pos, SWEEP_Z + 50.0] + ori
            self.robot.movel(above, vel=VELOCITY)
            self.robot.mwait()

            # 왼쪽 시작점 내려오기
            start = [cx - SWEEP_RANGE_X / 2, y_pos, SWEEP_Z] + ori
            self.robot.movel(start, vel=VELOCITY_SLOW)
            self.robot.mwait()

            # → 오른쪽 끝으로 밀기
            end = [cx + SWEEP_RANGE_X / 2, y_pos, SWEEP_Z] + ori
            self.robot.movel(end, vel=VELOCITY_SLOW)
            self.robot.mwait()

            # 들어올리기
            above_end = [cx + SWEEP_RANGE_X / 2, y_pos, SWEEP_Z + 50.0] + ori
            self.robot.movel(above_end, vel=VELOCITY)
            self.robot.mwait()

            progress = 10.0 + 80.0 * ((i + 1) / n_steps)
            fb("SWEEPING", progress)

        self._go_home()
        fb("DONE", 100.0)
        self.get_logger().info("[SWEEP] Complete.")
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

        approach  = list(target); approach[2]  = 530
        pick_pos  = list(target); pick_pos[2]  += PICK_Z_OFFSET
        bin_above = list(bin_pos); bin_above[2] += APPROACH_Z_OFFSET

        self._transit_to_work()

        # 1. APPROACH
        fb("APPROACH", 10.0)
        self.gripper.move(width=gp["width"], force=gp["force"])
        self.robot.movel(approach)
        self.robot.mwait()

        # 2. GRASP
        fb("GRASP", 30.0)
        self.robot.movel(pick_pos, vel=VELOCITY_SLOW)
        self.robot.mwait()
        self.gripper.close(force=gp["force"])

        # 파지 확인 + 재시도
        if not self.gripper.is_gripping():
            self.get_logger().warn("[PICK] No grip detected – retry.")
            self.robot.movel(approach)
            self.robot.mwait()
            self.robot.movel(pick_pos, vel=VELOCITY_SLOW)
            self.robot.mwait()
            self.gripper.close(force=min(gp["force"] + 100, 400))

        # 3. LIFT
        fb("LIFT", 50.0)
        self.robot.movel(approach)
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

    def _go_home(self, open_gripper: bool = True):
        self.robot.movej(J_HOME, vel=VELOCITY, acc=ACC)
        self.robot.mwait()
        if open_gripper:
            self.gripper.open()

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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
