import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
import DR_init
from dsr_msgs2.srv import MoveStop
from recycle_interfaces.action import MoveRobots


ROBOT_ID    = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id    = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
_dsr_node = rclpy.create_node("m0609_exec_dsr_init", namespace=ROBOT_ID)
DR_init.__dsr__node = _dsr_node

J_HOME = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]

#디텍팅 위치 
J_WORK = [-11.0, 26.0, 19.0, 0.0, 133.0, -12.0]  




class Stop_test(Node):
    def __init__(self):
        super().__init__('safety_control_node')
        
        self._stop_client = self.create_client(MoveStop, '/dsr01/motion/move_stop')
        self.get_logger().info("안전 제어 노드 초기화. 정지 서비스 통신망 구축 완료.")

        self._action_server = ActionServer(self,MoveRobots,"/recycle/exec/move_robot",
                                                execute_callback = self._execute_cb, goal_callback=self._goal_cb, cancel_callback=self._cancel_cb,
                                                callback_group = ReentrantCallbackGroup(),)
        self.get_logger().info("move_robot_server_ready")


        from DSR_ROBOT2 import movej
        movej(J_HOME,vel=50,acc=50)



    def trigger_async_stop(self):
        if not self._stop_client.service_is_ready():
            self.get_logger().error("정지 서비스가 현재 통신 불가 상태입니다.연결을 점검하십시오.")
            return

        request = MoveStop.Request()
        request.stop_mode = 2  # DR_SSTOP 부드럽게 정지!

        future = self._stop_client.call_async(request)
        
        self.get_logger().info("비동기 정지 명령 전송 완료. 즉시 제어권 반환됨.")



    def _stop_response_callback(self, future):
        """
        Doosan 제어기로부터 정지 명령 수신 결과가 도착했을 때만 비동기적으로 실행되는 콜백.
        """
        try:
            response = future.result()
            self.get_logger().info(f"정지 명령 서버 최종 응답: success={response.success}")
        except Exception as e:
            self.get_logger().error(f"정지 명령 응답 처리 중 치명적 예외 발생: {e}")




    def _execute_cb(self,goal_handle):
        goal = goal_handle.request
        move_mode = goal.move_mode 
        pos =  list(goal.goal_pos)
        posc = list(goal.goal_pos_c)
        vel =  goal.vel
        acc =  goal.acc 

        self.get_logger().info(f"로봇 이동 시작: {move_mode} / 목적지: {pos}")
        from DSR_ROBOT2 import amovej,check_motion
        if move_mode == "movej":
            amovej(pos,vel=20,acc=10)

     
        else:
            # 에러 처리
            pass

       

        while check_motion() != 0:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().warn("취소 명령 수신! 로봇 이동을 즉시 차단합니다.")
                return MoveRobots.Result(success=False, message="Canceled by BT")
            self.get_logger().info("동작중")
            time.sleep(0.1)
        
        time.sleep(0.5)
        goal_handle.succeed()
        self.get_logger().info("목표 지점 도달 완료.")
        return MoveRobots.Result(success=True, message="Reached")
        
    def _goal_cb(self,goal_request):
        self.get_logger().info("명령수용!")
        return GoalResponse.ACCEPT


    def _cancel_cb(self,cancel_request):
        self.get_logger().info("취소 명령 수용!")
        from DSR_ROBOT2 import check_motion
        self.trigger_async_stop()
        self.get_logger().info("멈춘다잉")
        while check_motion() != 0:
            time.sleep(0.1)
        self.get_logger().info("멈쳤다!")
        return CancelResponse.ACCEPT



def main(args=None):
    node     = Stop_test()
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