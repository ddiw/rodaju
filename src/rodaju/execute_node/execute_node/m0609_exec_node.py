#!/usr/bin/env python3
"""m0609_exec_node.py  ─  로봇 동작 수행 노드 (ROS 액션 서버).

실제 동작 로직은 execute_node.motions 모듈에 위임한다.
"""

import os

import numpy as np
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
import DR_init

T_GRIPPER2CAM_PATH = os.path.join(
    get_package_share_directory("execute_node"), "resource", "T_gripper2camera.npy"
)

from recycle_interfaces.action import PickPlace
from execute_node.constants import ROBOT_ID, ROBOT_MODEL, VELOCITY, ACC, J_HOME, J_WORK

DR_init.__dsr__id    = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
_dsr_node = rclpy.create_node("m0609_exec_dsr_init", namespace=ROBOT_ID)
DR_init.__dsr__node = _dsr_node

# DR_init 설정 이후 import
from execute_node.robot_api import RobotAPI, GripperAPI
from execute_node.motions import do_sweep, do_clean_desk, do_pick_place


class M0609ExecNode(Node):

    def __init__(self):
        super().__init__("m0609_exec_node")

        self.declare_parameter("gripper_angle_offset_deg", -90.0)
        self._angle_offset = self.get_parameter("gripper_angle_offset_deg").value

        self.robot   = RobotAPI()
        self.gripper = GripperAPI()
        self._busy   = False

        try:
            self._T_gripper2cam = np.load(T_GRIPPER2CAM_PATH)
            self.get_logger().info(f"Hand-Eye matrix loaded: {T_GRIPPER2CAM_PATH}")
        except Exception as e:
            self._T_gripper2cam = None
            self.get_logger().warn(f"T_gripper2camera.npy load failed: {e} → pixel estimate.")

        ActionServer(
            self, PickPlace, "/recycle/exec/pick_place",
            execute_callback = self._execute_cb,
            goal_callback    = self._goal_cb,
            cancel_callback  = self._cancel_cb,
            callback_group   = ReentrantCallbackGroup(),
        )
        self.get_logger().info("PickPlace action server ready.")
        self._go_home()
        self.get_logger().info("M0609ExecNode ready.")

    # ── 액션 서버 콜백 ────────────────────────────────────────

    def _goal_cb(self, _):
        if self._busy:
            self.get_logger().warn("Goal rejected – robot busy.")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, _):
        return CancelResponse.ACCEPT

    def _execute_cb(self, goal_handle):
        self._busy = True
        goal     = goal_handle.request
        feedback = PickPlace.Feedback()
        result   = PickPlace.Result()
        log      = self.get_logger()

        def fb(phase: str, progress: float):
            feedback.phase = phase; feedback.progress = progress
            goal_handle.publish_feedback(feedback)
            log.info(f"  [FB] {phase} {progress:.0f}%")

        try:
            cmd = goal.label
            if cmd == "SWEEP":
                success = do_sweep(self.robot, self.gripper, fb=fb, logger=log)
            elif cmd == "CLEAN_DESK":
                do_clean_desk(self.robot, self.gripper, logger=log); success = True
            elif cmd == "GOTO_WORK":
                self.robot.movej(J_WORK, vel=VELOCITY); self.robot.mwait(); success = True
            elif cmd == "HOME":
                self._go_home(); success = True
            else:
                success = do_pick_place(
                    self.robot, self.gripper, goal,
                    self._T_gripper2cam, self._angle_offset,
                    fb=fb, logger=log,
                )

            result.success = success
            result.message = "OK" if success else "FAILED"
            goal_handle.succeed() if success else goal_handle.abort()

        except Exception as e:
            log.error(f"Execute error: {e}")
            result.success = False; result.message = str(e)
            fb("ERROR", 0.0); self._go_home(); goal_handle.abort()
        finally:
            self._busy = False

        return result

    def _go_home(self):
        self.robot.movej(J_HOME, vel=VELOCITY, acc=ACC)
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
