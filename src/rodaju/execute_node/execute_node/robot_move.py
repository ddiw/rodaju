#!/usr/bin/env python3
import time
import sys

import threading

import rclpy
from rclpy.node import Node
import DR_init

from execute_node.onrobot import RG

# ----------------- Robot/Gripper Config -----------------
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

JREADY = [0, 0, 90, 0, 90, 0]
SPATULA_POS = [251.1, -201.0, 84.4, 129.1, -72.1, -149.1]  # task pose (posx)

GRIPPER_NAME = "rg2"
TOOLCHANGER_IP = "192.168.1.1"
TOOLCHANGER_PORT = "502"

# ----------------- DR_init -----------------
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("rokey_simple_move", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

# DSR 모션 함수가 ROS2 서비스/액션을 사용하므로 별도 스레드에서 spin
_executor = rclpy.executors.SingleThreadedExecutor()
_executor.add_node(dsr_node)
_spin_thread = threading.Thread(target=_executor.spin, daemon=True)
_spin_thread.start()

try:
    from DSR_ROBOT2 import movej, movel, mwait
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit(1)

# ----------------- Gripper -----------------
gripper = RG(GRIPPER_NAME, TOOLCHANGER_IP, TOOLCHANGER_PORT)


class RobotController(Node):
    def __init__(self):
        super().__init__("pick_and_place_onrobot")
        self.get_logger().info("Start pick & place with OnRobot RG gripper")

        # 초기 자세 + 그리퍼 오픈
        self.go_jready(open_grip=True)

        # 본 동작
        self.pick_and_place()

        # 마무리 복귀
        self.go_jready(open_grip=True)

    # ----------- Gripper Utils -----------
    def wait_gripper_idle(self, timeout=5.0, period=0.1):
        """
        busy 플래그(=get_status()[0])가 0이 될 때까지 대기
        """
        t0 = time.time()
        while True:
            st = gripper.get_status()  # [busy, grip, s1, ...]
            busy = st[0]
            if busy == 0:
                return True
            if time.time() - t0 > timeout:
                self.get_logger().warn("Gripper busy timeout")
                return False
            time.sleep(period)

    def gripper_open_blocking(self, force=400, timeout=5.0):
        """
        오픈 명령을 '확실히' 수행: idle 확인 -> 명령 -> idle 될 때까지 대기
        """
        self.wait_gripper_idle(timeout=timeout)
        gripper.open_gripper(force_val=force)
        return self.wait_gripper_idle(timeout=timeout)

    def gripper_close_blocking(self, force=400, timeout=5.0):
        """
        클로즈 명령을 '확실히' 수행
        """
        self.wait_gripper_idle(timeout=timeout)
        gripper.close_gripper(force_val=force)
        ok = self.wait_gripper_idle(timeout=timeout)

        # 잡힘 여부도 같이 보고 싶으면 여기서 확인 가능 (옵션)
        st = gripper.get_status()
        if st[1] == 1:
            self.get_logger().info("Grip detected ✅")
        else:
            self.get_logger().warn("Grip NOT detected (may be empty or slip) ⚠️")

        return ok

    # ----------- Robot Motion Utils -----------
    def go_jready(self, open_grip=False):
        movej(JREADY, vel=VELOCITY, acc=ACC)
        mwait()
        if open_grip:
            self.gripper_open_blocking(force=400, timeout=5.0)

    # ----------- Main Task -----------
    def pick_and_place(self):
        self.get_logger().info(f"Target (SPATULA_POS): {SPATULA_POS}")

        # 1) 대상 위치로 이동
        movel(SPATULA_POS, vel=VELOCITY, acc=ACC)
        mwait()

        # 2) 집기
        self.gripper_close_blocking(force=400, timeout=5.0)

        # 3) 원위치(JREADY) 복귀 (집은 채)
        self.go_jready(open_grip=False)

        # 4) 다시 같은 위치로 이동 (내려놓기 위치)
        movel(SPATULA_POS, vel=VELOCITY, acc=ACC)
        mwait()

        # 5) 내려놓기
        self.gripper_open_blocking(force=400, timeout=5.0)


def main(args=None):
    node = RobotController()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()