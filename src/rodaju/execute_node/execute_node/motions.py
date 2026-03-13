#!/usr/bin/env python3
"""로봇 동작 시퀀스 (ROS 노드와 분리된 pure 로직).

모든 함수는 robot(RobotAPI), gripper(GripperAPI) 를 받아 동작을 수행한다.
"""

import time

from execute_node.constants import (
    VELOCITY, ACC, VELOCITY_SLOW,
    J_HOME, J_WORK,
    GATHER_LIFT_Z, GATHER_STEPS, CLEAN_STEPS,
    BROOM_HOLDER_POS, BROOM_APPROACH_Z,
    APPROACH_Z_OFFSET, PICK_Z_OFFSET, MAX_GRASP_RETRIES,
    BIN_POSITIONS, GRIPPER_PARAMS,
)
from execute_node.coord_transform import cam_to_robot, pixel_estimate
from execute_node.shake_classifier import do_shake_classify

# ═══════════════════════════════════════════════════════════════
#  빗자루 공통 헬퍼
# ═══════════════════════════════════════════════════════════════

def _pick_broom(robot, gripper) -> list:
    """거치대에서 빗자루를 집어 J_WORK로 이동. broom_above 좌표 반환."""
    broom_above = list(BROOM_HOLDER_POS)
    broom_above[2] += BROOM_APPROACH_Z

    gripper.open()
    robot.movel(broom_above, vel=VELOCITY_SLOW, acc=ACC);   robot.mwait()
    robot.movel(BROOM_HOLDER_POS, vel=VELOCITY_SLOW, acc=ACC); robot.mwait()
    gripper.close(force=GRIPPER_PARAMS["DEFAULT"]["force"])
    robot.movel(broom_above, vel=VELOCITY_SLOW, acc=ACC);   robot.mwait()
    robot.movej(J_WORK, vel=VELOCITY_SLOW, acc=ACC);        robot.mwait()
    return broom_above


def _return_broom(robot, gripper, broom_above: list):
    """J_WORK → 거치대로 빗자루 반납."""
    robot.movej(J_WORK, vel=VELOCITY, acc=ACC);              robot.mwait()
    robot.movel(broom_above, vel=VELOCITY_SLOW, acc=ACC);    robot.mwait()
    robot.movel(BROOM_HOLDER_POS, vel=VELOCITY_SLOW, acc=ACC); robot.mwait()
    gripper.open()
    robot.movel(broom_above, vel=VELOCITY, acc=ACC);         robot.mwait()


def _run_steps(robot, steps: list):
    """(start, end) 쌍 리스트를 순서대로 훑는다 (GATHER_LIFT_Z 만큼 들어올리며 이동)."""
    for start, end in steps:
        start_above = list(start); start_above[2] += GATHER_LIFT_Z
        end_above   = list(end);   end_above[2]   += GATHER_LIFT_Z

        robot.movel(start_above, vel=VELOCITY, acc=ACC); robot.mwait()
        robot.movel(start,       vel=VELOCITY, acc=ACC); robot.mwait()
        robot.movel(end,         vel=VELOCITY, acc=ACC); robot.mwait()
        robot.movel(end_above,   vel=VELOCITY, acc=ACC); robot.mwait()


# ═══════════════════════════════════════════════════════════════
#  공개 동작 함수
# ═══════════════════════════════════════════════════════════════

def do_sweep(robot, gripper, fb=None, logger=None) -> bool:
    """GATHER_STEPS 순서대로 쓰레기를 모은 뒤 빗자루를 반납하고 홈 복귀.

    Args:
        robot:   RobotAPI
        gripper: GripperAPI
        fb:      feedback 콜백 (phase: str, progress: float) → None
        logger:  rclpy logger (optional)
    """
    _fb = fb or (lambda p, v: None)
    _fb("SWEEP_START", 5.0)
    if logger: logger.info("[SWEEP] Starting gather sequence.")

    broom_above = _pick_broom(robot, gripper)
    _fb("BROOM_PICK", 10.0)

    n = len(GATHER_STEPS)
    for i, step in enumerate(GATHER_STEPS):
        _run_steps(robot, [step])
        _fb("GATHERING", 15.0 + 70.0 * ((i + 1) / n))

    _fb("BROOM_RETURN", 90.0)
    _return_broom(robot, gripper, broom_above)

    robot.movej(J_HOME, vel=VELOCITY, acc=ACC); robot.mwait()
    _fb("DONE", 100.0)
    if logger: logger.info("[SWEEP] Gather complete.")
    return True


def do_clean_desk(robot, gripper, logger=None):
    """CLEAN_STEPS 순서대로 쓰레기를 테이블 끝으로 쓸어내고 빗자루를 반납."""
    if logger: logger.info("[CLEAN] Starting clean desk.")
    broom_above = _pick_broom(robot, gripper)
    _run_steps(robot, CLEAN_STEPS)
    _return_broom(robot, gripper, broom_above)
    if logger: logger.info("[CLEAN] Clean desk complete.")


def do_pick_place(robot, gripper, goal, T_gripper2cam, angle_offset: float,
                  fb=None, logger=None) -> bool:
    """접근 → 파지(재시도) → 분류함 이동 → 투입 → 홈 복귀.

    Args:
        robot, gripper:  하드웨어 API
        goal:            PickPlace.Goal (ROS action goal)
        T_gripper2cam:   Hand-Eye 캘리브레이션 행렬 (None이면 픽셀 추정)
        angle_offset:    그리퍼 RZ 보정각 (deg)
        fb:              feedback 콜백 (phase, progress) → None
        logger:          rclpy logger (optional)
    """
    _fb = fb or (lambda p, v: None)

    bin_id  = goal.bin_id
    gp      = GRIPPER_PARAMS.get(bin_id, GRIPPER_PARAMS["DEFAULT"])
    bin_pos = BIN_POSITIONS.get(bin_id, BIN_POSITIONS["BIN_PLASTIC_EMPTY"])

    # ── 목표 좌표 계산 ────────────────────────────────────
    robot_posx = robot.get_posx()
    if goal.has_3d and T_gripper2cam is not None:
        target = cam_to_robot(
            goal.pick_x_m, goal.pick_y_m, goal.pick_z_m,
            T_gripper2cam, robot_posx, logger=logger,
        )
    else:
        if T_gripper2cam is None and logger:
            logger.warn("[CAM2ROBOT] No calibration matrix – using pixel estimate.")
        target = pixel_estimate(goal.pick_cx, goal.pick_cy, robot_posx, logger=logger)

    target[5] = (goal.pick_angle_deg + angle_offset) % 360.0
    if logger:
        logger.info(
            f"[PICK] label={goal.label} rz={target[5]:.1f}°  "
            f"target=({target[0]:.1f},{target[1]:.1f},{target[2]:.1f})mm"
        )

    approach  = list(target);  approach[2]  += APPROACH_Z_OFFSET
    pick_pos  = list(target);  pick_pos[2]  += PICK_Z_OFFSET
    bin_above = list(bin_pos); bin_above[2] += APPROACH_Z_OFFSET

    # ── 1. APPROACH ───────────────────────────────────────
    robot.movej(J_WORK); robot.mwait()   # singularity 방지
    _fb("APPROACH", 10.0)
    gripper.move(width=gp["width"], force=gp["force"])
    robot.movel(approach); robot.mwait()

    # ── 2. GRASP (재시도) ──────────────────────────────────
    gripped = False
    for attempt in range(1, MAX_GRASP_RETRIES + 1):
        _fb("GRASP", 30.0)
        robot.movel(pick_pos, vel=VELOCITY_SLOW); robot.mwait()
        gripper.close(force=gp["force"])
        if gripper.is_gripping():
            gripped = True
            break
        if logger:
            logger.warn(f"[GRASP] attempt {attempt}/{MAX_GRASP_RETRIES} – retry.")
        robot.movel(approach, vel=VELOCITY_SLOW); robot.mwait()
        gripper.move(width=gp["width"], force=gp["force"])

    if not gripped:
        if logger: logger.error("[GRASP] Failed after all retries.")
        robot.movel(approach); robot.mwait()
        return False

    # ── 3. LIFT & 4. MOVE_BIN ─────────────────────────────
    _fb("LIFT", 50.0)
    robot.movel(approach);  robot.mwait()

    if bin_id == "BIN_PLASTIC":
        robot.movej(J_HOME, vel=VELOCITY, acc=ACC); robot.mwait()
        bin_id    = do_shake_classify(robot, fb=_fb, logger=logger)
        bin_pos   = BIN_POSITIONS[bin_id]
        bin_above = list(bin_pos); bin_above[2] += APPROACH_Z_OFFSET

    _fb("MOVE_HOME", 60.0)
    robot.movej(J_HOME);    robot.mwait()
    _fb("MOVE_BIN", 65.0)
    robot.movel(bin_above); robot.mwait()

    # ── 5. PLACE & 6. RETURN ──────────────────────────────
    _fb("PLACE", 85.0)
    robot.movel(bin_pos, vel=VELOCITY_SLOW); robot.mwait()
    gripper.open()
    time.sleep(0.3)

    _fb("RETURN", 95.0)
    robot.movel(bin_above); robot.mwait()
    robot.movej(J_HOME, vel=VELOCITY, acc=ACC); robot.mwait()

    _fb("DONE", 100.0)
    if logger: logger.info(f"[PICK_PLACE] {goal.label} → {bin_id} complete.")
    return True
