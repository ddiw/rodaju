#!/usr/bin/env python3
"""카메라 → 로봇 좌표 변환 유틸리티 (free function)."""

import numpy as np
from scipy.spatial.transform import Rotation

from execute_node.constants import TABLE_CENTER_POS


def posx_to_matrix(posx: list) -> np.ndarray:
    """DSR get_current_posx() 결과 [x,y,z,rx,ry,rz] (mm, deg) → 4×4 동차 변환행렬."""
    x, y, z, rx, ry, rz = posx
    R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = [x, y, z]
    return T


def cam_to_robot(
    x_m: float, y_m: float, z_m: float,
    T_gripper2cam: np.ndarray,
    robot_posx: list,
    logger=None,
) -> list:
    """카메라 좌표계(m) → 로봇 베이스 좌표계(mm) 변환.

    Args:
        x_m, y_m, z_m: 카메라 좌표 (m)
        T_gripper2cam:  Hand-Eye 캘리브레이션 행렬 (4×4)
        robot_posx:     현재 TCP 포즈 [x,y,z,rx,ry,rz]
        logger:         rclpy logger (optional)
    """
    coord = np.append(
        np.array([x_m * 1000.0, y_m * 1000.0, z_m * 1000.0]),
        1.0,
    )
    base2gripper = posx_to_matrix(robot_posx)
    td_coord = np.dot(base2gripper @ T_gripper2cam, coord)

    DEPTH_OFFSET = -5.0
    MIN_Z        =  2.0
    if td_coord[2] and np.sum(td_coord[:3]) != 0:
        td_coord[2] += DEPTH_OFFSET
        td_coord[2]  = max(td_coord[2], MIN_Z)

    target_pos = list(td_coord[:3]) + list(robot_posx[3:])

    if logger:
        logger.debug(
            f"[CAM2ROBOT] cam=({x_m*1000:.1f},{y_m*1000:.1f},{z_m*1000:.1f})mm "
            f"-> base=({td_coord[0]:.1f},{td_coord[1]:.1f},{td_coord[2]:.1f})mm"
        )
    return target_pos


def pixel_estimate(cx: int, cy: int, robot_posx: list, logger=None) -> list:
    """Depth 없을 때 픽셀 기반 fallback 좌표 추정."""
    SCALE = 0.5
    pos = list(robot_posx)
    pos[0] += (cx - 320) * SCALE
    pos[1] += (cy - 240) * SCALE
    pos[2]  = TABLE_CENTER_POS[2]
    if logger:
        logger.warn(f"[ESTIMATE] pixel fallback cx={cx} cy={cy}")
    return pos
