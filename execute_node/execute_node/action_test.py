import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from recycle_interfaces.action import MoveRobots

class StopActionTest(Node):
    def __init__(self):
        super().__init__("stop_test_action")
        self._action_client = ActionClient(
                self, MoveRobots, "/recycle/exec/move_robot",
                callback_group=ReentrantCallbackGroup())
        
        # 취소 명령에 필요한 골 핸들과 1회성 타이머 객체 초기화
        self._goal_handle = None
        self._cancel_timer = None

    def send_test_goal(self):
        self.get_logger().info("액션 서버 대기 중...")
        self._action_client.wait_for_server()

        goal_msg = MoveRobots.Goal()
        goal_msg.move_mode = "movej"
        goal_msg.goal_pos = [-11.0, 26.0, 19.0, 0.0, 133.0, -12.0]  

        self.get_logger().info("목표(Goal) 전송 완료")
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().error("서버에서 목표를 거부했습니다.")
            return

        self.get_logger().info("목표가 수락되었습니다. 2초 후 취소 시퀀스를 가동합니다.")
        
        # 메인 스레드를 블로킹하지 않고 2초 뒤 _cancel_goal_callback을 실행하는 타이머 생성
        self._cancel_timer = self.create_timer(2.0, self._cancel_goal_callback)
        

    def re_action_callback(self):
        self.re_action_timer.cancel()
        
        self.get_logger().warning("다시 액션 실행....")
        self._action_client.wait_for_server()

        goal_msg = MoveRobots.Goal()
        goal_msg.move_mode = "movej"
        goal_msg.goal_pos = [-11.0, 26.0, 19.0, 0.0, 133.0, -12.0]  
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self._goal_cb)
        self.get_logger().info("목표(Goal) 전송 완료")

    def _goal_cb(self, future):
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().error("서버에서 목표를 거부했습니다.")
            return
        self.get_logger().info("목표로 이동합니당")
  
        

    def _cancel_goal_callback(self):
        # 타이머는 1회성 이벤트이므로 실행 즉시 해제하여 반복 호출을 방지합니다.
        self._cancel_timer.cancel()

        if self._goal_handle is None:
            return

        self.get_logger().warning("목표 취소(Cancel) 요청 전송 중...")
        cancel_future = self._goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self._cancel_response_callback)
        self.re_action_timer = self.create_timer(3.0, self.re_action_callback)



    def _cancel_response_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info("취소 요청이 서버에 정상적으로 수용되었습니다.")
        else:
            self.get_logger().error("서버가 취소 요청을 거부했거나 처리하지 못했습니다.")

def main(args=None):
    rclpy.init(args=args)
    action_client_node = StopActionTest()
    
    # 목표 전송 트리거
    action_client_node.send_test_goal()

    try:
        rclpy.spin(action_client_node)
    except KeyboardInterrupt:
        pass
    finally:
        action_client_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
        



        