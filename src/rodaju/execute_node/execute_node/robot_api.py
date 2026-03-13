#!/usr/bin/env python3
"""DSR 로봇 / RG 그리퍼 API 래퍼.

DSR_ROBOT2는 DR_init 설정 이후에만 import 가능하므로
RobotAPI.__init__ 내에서 지연 import를 사용한다.
"""

import time

from execute_node.onrobot import RG
from execute_node.constants import VELOCITY, ACC


class RobotAPI:
    """DSR 로봇 API 래퍼."""

    def __init__(self):
        from DSR_ROBOT2 import movej, movel, get_current_posx, mwait, trans, get_external_torque
        self._get_ext_torque = get_external_torque
        self._movej           = movej
        self._movel           = movel
        self._get_current_posx = get_current_posx
        self._mwait           = mwait
        self._trans           = trans

    def movej(self, joints, vel=VELOCITY, acc=ACC, radius=0.0):
        self._movej(joints, vel=vel, acc=acc, radius=radius)

    def movel(self, pos, vel=VELOCITY, acc=ACC):
        self._movel(pos, vel=vel, acc=acc)

    def mwait(self):
        self._mwait()

    def get_posx(self) -> list:
        return list(self._get_current_posx()[0])

    def trans_offset(self, pos: list, delta: list) -> list:
        return list(self._trans(pos, delta))
    
    def get_external_torque(self) -> list:
        return list(self._get_ext_torque())

class GripperAPI:
    """OnRobot RG2 그리퍼 API 래퍼."""

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
        # 1단계: busy=1 (동작 시작) 대기 (최대 0.5초)
        t = time.time()
        while time.time() - t < 0.5:
            if self._g.get_status()[0]:
                break
            time.sleep(0.05)
        # 2단계: busy=0 (동작 완료) 대기
        t = time.time()
        while time.time() - t < timeout:
            if not self._g.get_status()[0]:
                break
            time.sleep(0.1)
