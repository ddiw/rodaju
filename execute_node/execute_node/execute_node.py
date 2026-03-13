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
from dsr_msgs2.srv import MoveStop
# ── Hand-Eye 캘리브레이션 파일 경로 ──────────────────────────────
T_GRIPPER2CAM_PATH = os.path.join(
    get_package_share_directory("execute_node"), "resource", "T_gripper2camera.npy"
)

try:
    from recycle_interfaces.action import MoveRobots,ControlGripper
    INTERFACES_AVAILABLE = True
except ImportError:
    INTERFACES_AVAILABLE = False

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
    from DSR_ROBOT2 import movej,movel,get_current_posx,posx,posj,amovej,amovel,amovejx,DR_MV_MOD_REL,DR_BASE
    DSR_AVAILABLE = True
except ImportError:
    DSR_AVAILABLE = False
    print("[WARN] DSR_ROBOT2 not available - simulation mode.")

try:
    from execute_node.onrobot import RG
    GRIPPER_AVAILABLE = True
except ImportError:
    GRIPPER_AVAILABLE = False
    print("[WARN] OnRobot not available - simulation mode.")



#대기 상태 위치 (standby)  
J_HOME = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]

#디텍팅 위치 
J_WORK = [-11.0, 26.0, 19.0, 0.0, 133.0, -12.0]  



APPROACH_Z_OFFSET = 80.0
PICK_Z_OFFSET     = 0.0



T_GRIPPER2CAM_PATH = os.path.join(get_package_share_directory("execute_node"), "resource", "T_gripper2camera.npy")


class ExecudeNode(Node):
    def __init__(self):
        super().__init__("execute_node")

        self.declare_parameter("gripper_angle_offset_deg", -90.0)
        self._gripper_angle_offset = self.get_parameter("gripper_angle_offset_deg").value

        self._lock   = threading.Lock()

        try:
            self._T_gripper2cam = np.load(T_GRIPPER2CAM_PATH)
            self.get_logger().info(f"Hand-Eye matrix loaded: {T_GRIPPER2CAM_PATH}\n {self._T_gripper2cam}")
        except Exception as e:
            self._T_gripper2cam = None
            self.get_logger().warn(f"T_gripper2camera.npy load failed: {e}\n -> falling back to pixel estimate.")
        
        #초기 홈 위치 이동 
        movej(posj(J_HOME),vel=100,acc=100)
        self.rg = None
        
        if GRIPPER_AVAILABLE:
            try:
                self.rg = RG("rg2", "192.168.1.1", 502)
            except Exception as e:
                print(f"[WARN] Gripper init failed: {e}")


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

        #정지 서비스 클라이언트
        self._stop_client = self.create_client(MoveStop, '/dsr01/motion/move_stop')
        self.get_logger().info("정지 서비스 준비완료!")


        #액션 서버 
        if INTERFACES_AVAILABLE:
            self._action_server = ActionServer(self,MoveRobots,"/recycle/exec/move_robot",
                                                execute_callback = self._execute_cb, goal_callback=self._goal_cb, cancel_callback=self._cancel_cb,
                                                callback_group = ReentrantCallbackGroup(),)
            self.get_logger().info("move_robot_server_ready")
            self.g_action_server = ActionServer(self,ControlGripper,"/recycle/exec/control_gripper",
                                                execute_callback = self.g_exec_cb, goal_callback=self.g_goal_cb, cancel_callback=self.g_canc_cb,
                                                callback_group = ReentrantCallbackGroup(),)
            self.get_logger().info("control_gripper_server_ready")

        else:
            self.get_logger().warn("recycle_interfaces not found - action server disabled.")

    def trigger_async_stop(self):
        if not self._stop_client.service_is_ready():
            self.get_logger().error("정지 서비스가 현재 통신 불가 상태입니다.연결을 점검하십시오.")
            return

        request = MoveStop.Request()
        request.stop_mode = 2  # DR_SSTOP 부드럽게 정지!

        future = self._stop_client.call_async(request)
        
        self.get_logger().info("비동기 정지 명령 전송 완료. 즉시 제어권 반환됨.")


    def _execute_cb(self,goal_handle):
        goal = goal_handle.request
        move_mode = goal.move_mode 
        pos =  list(goal.goal_pos)
        vel =  goal.vel
        acc =  goal.acc 

        self.get_logger().info(f"로봇 이동 시작: {move_mode} / 목적지: {pos}")

        if move_mode == "movej":
            amovej(pos,vel=vel,acc=acc)

        elif  move_mode == "movel":
            amovel(pos,vel=vel,acc=acc)


        elif move_mode == "lift":
            movel(posx(0,0,100,0,0,0),vel=30,acc=30,mod=DR_MV_MOD_REL)

        

        else:
            x_m, y_m, z_m, angle_deg = pos[0], pos[1], pos[2], pos[3]
            target_abs_pos = self._cam_to_robot(x_m, y_m, z_m)

            if target_abs_pos == [0.0]*6:
                goal_handle.abort()
                return MoveRobots.Result(success=False, message="Matrix Error")
            
            if move_mode == "apporach":target_abs_pos[2] = 200.0 
            target_abs_pos[5] += angle_deg # 물체가 돌아간 만큼 그리퍼도 돌립니다.
            
            self.get_logger().info(f"변환된 최종 타겟 절대 좌표: {target_abs_pos}")
            self.get_logger().info(f"객체 상단으로 이동 x: {target_abs_pos[0]} y: {target_abs_pos[1]} z: {target_abs_pos[2]}")
            amovel(target_abs_pos, vel=vel, acc=acc)
        

        from DSR_ROBOT2 import check_motion, DR_SSTOP, drl_script_stop

        while check_motion() != 0:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().warn("취소 명령 수신! 로봇 이동을 즉시 차단합니다.")
                return MoveRobots.Result(success=False, message="Canceled by BT")

            time.sleep(0.05)
        
        time.sleep(0.5)
        goal_handle.succeed()
        self.get_logger().info("목표 지점 도달 완료.")
        return MoveRobots.Result(success=True, message="Reached")
        
    def _goal_cb(self,goal_request):
        self.get_logger().info("명령수용!")
        return GoalResponse.ACCEPT


    def _cancel_cb(self,cancel_request):
        self.get_logger().info("취소 명령 수용!")
        self.trigger_async_stop()
        
        return CancelResponse.ACCEPT


    def g_exec_cb(self,goal_handle):
        #그리퍼 제어하는 코드 
        goal = goal_handle.request
        command = goal.command
        force =   goal.force

        
        if (command == "close"):
            self.rg.close_gripper(force_val=force)
        elif (command == "open"):
            self.rg.open_gripper(force_val=force)

       

        if self.rg is None:
            self.get_logger().error("그리퍼 객체가 없습니다. 통신 불가.")
            goal_handle.abort()
            return ControlGripper.Result(success=False, message="Not connected")
        
        while True:
            status = self.rg.get_status()
            is_busy = status[0]
            if goal_handle.is_cancel_requested:
                self.rg.set_control_mode(8) 
                goal_handle.canceled()
                self.get_logger().warn("취소 명령 수신! 그리퍼 동작 즉시 차단 완료.")
                return ControlGripper.Result(success=False, message="Canceled by BT")
            if is_busy == 0: break
            time.sleep(0.05)

        final_status = self.rg.get_status()
        grip_detected = final_status[1]
        goal_handle.succeed()

        time.sleep(0.5)

        if command == "close" and grip_detected == 1:
            self.get_logger().info("물체 파지 감지(Grip Detected) 및 동작 완료.")
        else:
            self.get_logger().info("그리퍼 동작 완료.")
            
        return ControlGripper.Result(success=True, message="Completed")    
      

    def g_goal_cb(self,goal_request):
        self.get_logger().info("명령수용!")
        return GoalResponse.ACCEPT

    def g_canc_cb(self,cancel_request):
        self.get_logger().info("취소 명령 수용!")
        return CancelResponse.ACCEPT
    

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

        # 카메라 좌표 m -> mm
        coord = np.append(
            np.array([x_m * 1000.0, y_m * 1000.0, z_m * 1000.0]),
            1.0
        )

        # 현재 TCP 포즈 -> T_base2gripper
        robot_posx    = list(get_current_posx()[0])
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

def main(args=None):
    node     = ExecudeNode()
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



if __name__=="__main__":
    main()