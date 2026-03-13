import time
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
import py_trees

# 메시지 및 액션 인터페이스 (본인 환경에 맞게 확인 필수)
from recycle_interfaces.msg import Detections2D
from recycle_interfaces.action import PickPlace

# =====================================================================
# 1. 시각/판단 BT 노드 (DetectAndPrioritize)
# =====================================================================
class DetectAndPrioritize(py_trees.behaviour.Behaviour):
    def __init__(self, name="Detect_Trash"):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(key="detections", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="last_vision_time", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_goal", access=py_trees.common.Access.WRITE)
        
        self.priority = ['paper_cup', 'can', 'pet']
        self.exclude = []

    def update(self):
        current_time = time.time()
        
        # 타임아웃 검증: 1초 이상 갱신되지 않은 데이터는 폐기 (노이즈 방지)
        try:
            last_time = self.blackboard.last_vision_time
            if current_time - last_time > 1.0:
                return py_trees.common.Status.FAILURE
        except KeyError:
            return py_trees.common.Status.FAILURE

        try:
            dets = self.blackboard.detections
        except KeyError:
            return py_trees.common.Status.FAILURE

        if not dets:
            return py_trees.common.Status.FAILURE

        # 무결성 검증 및 필터링
        indexes = [x for x in self.priority if x not in self.exclude]
        valid_det = [
            d for d in dets 
            if d.label in indexes 
            and d.has_3d 
            and d.z_m > 0.01 
            and d.x_m is not None
        ]

        if not valid_det:
            return py_trees.common.Status.FAILURE

        # 람다 정렬 (우선순위 -> 최단 거리)
        valid_det.sort(key=lambda x: (indexes.index(x.label), math.hypot(x.x_m, x.y_m)))

        # 최적 타겟 블랙보드 기록
        self.blackboard.target_goal = valid_det[0]
        
        return py_trees.common.Status.SUCCESS


# =====================================================================
# 2. 근육/실행 BT 노드 (ExecutePickPlace)
# =====================================================================
class ExecutePickPlace(py_trees.behaviour.Behaviour):
    def __init__(self, name="Execute_Robot", ros_node=None):
        super().__init__(name)
        self.node = ros_node
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(key="target_goal", access=py_trees.common.Access.READ)
        
        self.action_client = ActionClient(self.node, PickPlace, "/recycle/exec/pick_place")
        self.action_status = "IDLE"

    def initialise(self):
        """노드가 처음 실행되거나, 이전 사이클이 종료된 후 재진입 시 호출"""
        self.action_status = "WAITING_ACCEPT"
        
        if not self.action_client.server_is_ready():
            self.node.get_logger().error("액션 서버 오프라인.")
            self.action_status = "FAILURE"
            return

        target = self.blackboard.target_goal

        goal_msg = PickPlace.Goal()
        goal_msg.label = target.label
        goal_msg.has_3d = target.has_3d
        goal_msg.pick_x_m = target.x_m
        goal_msg.pick_y_m = target.y_m
        goal_msg.pick_z_m = target.z_m

        self.node.get_logger().info(f"작업 명령 전송: {target.label} (X:{target.x_m:.2f}, Y:{target.y_m:.2f})")
        
        # 비동기 통신: 트리의 실행을 차단(Block)하지 않음
        future = self.action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def update(self):
        """0.1초마다 트리에 의해 호출되어 현재 상태 반환"""
        if self.action_status == "SUCCESS":
            return py_trees.common.Status.SUCCESS
        elif self.action_status == "FAILURE":
            return py_trees.common.Status.FAILURE
        else:
            # 상태가 WAITING_ACCEPT 이거나 RUNNING일 경우 대기
            return py_trees.common.Status.RUNNING

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error("액션 서버에서 명령 거부됨.")
            self.action_status = "FAILURE"
            return

        self.action_status = "RUNNING"
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.success:
            self.node.get_logger().info("작업 완료 성공.")
            self.action_status = "SUCCESS"
        else:
            self.node.get_logger().error("작업 완료 실패.")
            self.action_status = "FAILURE"


# =====================================================================
# 3. 메인 ROS2 뼈대 노드 (ManageNode)
# =====================================================================
class ManageNode(Node):
    def __init__(self):
        super().__init__("manage_node")
        
        # 블랙보드 초기화 및 데이터 격리 구역 생성
        self.blackboard = py_trees.blackboard.Client(name="ManageNode_Board")
        self.blackboard.register_key(key="detections", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="last_vision_time", access=py_trees.common.Access.WRITE)

        # 비전 데이터 구독
        self.create_subscription(
            Detections2D, 
            "/recycle/vision/detections", 
            self.vision_callback, 
            10
        )

        # 행동 트리 아키텍처 조립
        self.tree = self.create_behavior_tree()

        # 트리 실행용 타이머 (10Hz)
        self.timer = self.create_timer(0.1, self.tick_tree)
        self.get_logger().info("시스템 초기화 완료. BT 틱 발생 시작.")

    def vision_callback(self, msg):
        """데이터 판단 배제. 오직 블랙보드 기록만 수행."""
        self.blackboard.detections = msg.detections
        self.blackboard.last_vision_time = time.time()

    def create_behavior_tree(self):
        """시퀀스(Sequence)와 셀렉터(Selector)를 이용한 흐름 제어망 구축"""
        root = py_trees.composites.Selector(name="Main_Root", memory=False)
        
        # memory=True 속성으로 인해 이전 작업(RUNNING) 상태를 기억함
        cycle = py_trees.composites.Sequence(name="Pick_Place_Cycle", memory=True)
        
        detect_node = DetectAndPrioritize(name="Detect_Trash")
        execute_node = ExecutePickPlace(name="Execute_Robot", ros_node=self)
        
        cycle.add_children([detect_node, execute_node])
        
        # 쓰레기가 없어 cycle이 FAILURE를 반환하면 실행되는 대기 상태
        standby_node = py_trees.behaviours.Success(name="Standby")
        
        root.add_children([cycle, standby_node])
        return py_trees.trees.BehaviourTree(root)

    def tick_tree(self):
        """타이머에 의해 0.1초마다 트리를 강제 순회함"""
        self.tree.tick()


def main(args=None):
    rclpy.init(args=args)
    node = ManageNode()
    
    # [비판적 요소] 비동기 콜백(액션 처리)과 타이머(틱 발생)의 충돌(Deadlock) 방지를 위해
    # 반드시 MultiThreadedExecutor를 사용해야 합니다.
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()