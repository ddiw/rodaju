import math
import py_trees
import py_trees_ros
from py_trees.common import Status, Access
import rclpy
from recycle_interfaces.msg import SortCommand, Detections2D
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from recycle_interfaces.action import PickPlace,Detectinginit




class CalculateTarget(py_trees.behaviour.Behaviour):
    def __init__(self, name="CalculateTarget"):
        super().__init__(name)

        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(key="vision_data",access=Access.READ)
        self.blackboard.register_key(key="target",access=Access.WRITE)
        self.blackboard.register_key(key="priority_exclude",access=Access.READ)

    def update(self):
        if not self.blackboard.exists("vision_data") or not self.blackboard.exists("priority_exclude"):
            return Status.FAILURE
        target_list = self.blackboard.vision_data
        cmd_msg = self.blackboard.priority_exclude
        
        # 2. 메시지 필드명에 맞춘 정확한 데이터 추출 (SortCommand 구조 기준)
        priority = cmd_msg.priority_order 
        exclude = cmd_msg.exclude_mask

        filltered_priority= [i  for i in priority if i not in priority]
        self.priority = priority + filltered_priority
        
        cls = [i for i in self.priority if i not in exclude] #클래스 필터 
        valid_det = [i for i in target_list.detections if i.label in cls]

        if not valid_det:
            self.logger.info("유효한 타겟이 없습니다.")
            return Status.FAILURE
        
        targets_sort = sorted(valid_det, key = lambda x: (cls.index(x.label), math.hypot(x.x_m, x.y_m)))
        target = targets_sort[0]

        self.blackboard.target = target
        self.logger.info(f"타겟 선정! : label: {target.label}  x: {target.x_m} y: {target.y_m} z: {target.z_m} ")

        return Status.SUCCESS
    

class GetGoalPose(py_trees.behaviour.Behaviour):
    def __init__(self, name="GetGoalPose"):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(key="target",access=Access.READ)
        self.blackboard.register_key(key="goal_pos",access=Access.WRITE)

    def update(self):
        if not self.blackboard.exists("target") :
            return Status.FAILURE
        target = self.blackboard.target

        goal_pos = PickPlace.Goal()
        goal_pos.label = target.label
        goal_pos.has_3d = target.has_3d
        goal_pos.pick_x_m = target.x_m
        goal_pos.pick_y_m = target.y_m
        goal_pos.pick_z_m = target.z_m
        goal_pos.bin_id = f"BIN_{target.label}"

        self.blackboard.goal_pos = goal_pos
        self.logger.info(f"목표 생성 완료: {goal_pos.label} (X_m: {goal_pos.pick_x_m:.3f}) (Y_m: {goal_pos.pick_y_m:.3f}) (Z_m: {goal_pos.pick_z_m:.3f})")

        return Status.SUCCESS
    

def main():
    rclpy.init()
    bt_node = rclpy.create_node("main_node")
    root = py_trees.composites.Sequence(name="Data_tree",memory=True)

    default_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE
    )

    calculate_target = CalculateTarget(name="CalculateTarget")
    get_goal = GetGoalPose(name="GetGoalPose")

    cmd_to_black_board = py_trees_ros.subscribers.ToBlackboard(
        name="cmd_to_black_board",
        topic_name="/recycle/command",  
        qos_profile=default_qos,  
        topic_type=SortCommand,
        blackboard_variables={"priority_exclude": None},
        clearing_policy=py_trees.common.ClearingPolicy.NEVER,  
    )

    detect_to_black_board = py_trees_ros.subscribers.ToBlackboard(
        name = "detect_to_black_board",
        topic_name = "/recycle/vision/detections",
        qos_profile = default_qos,
        topic_type=Detections2D,
        blackboard_variables={"vision_data":None},
        clearing_policy=py_trees.common.ClearingPolicy.NEVER,
    )

    pick_action_node = py_trees_ros.action_clients.FromBlackboard(
        name = "PickActionClient",
        action_type=PickPlace,
        action_name="/recycle/exec/pick_place",
        key = "goal_pos")
    
    detect_pose_action_node = py_trees_ros.action_clients.FromBlackboard(
        name = "detect_pose",
        action_type=Detectinginit,
        action_name="recycle/exec/detectpose",
        key = "det_pos")
    
    root.add_children([cmd_to_black_board,detect_to_black_board,calculate_target,get_goal])
    
    tree = py_trees_ros.trees.BehaviourTree(root)
    try:
        tree.setup(node=bt_node)
    except Exception as e:
        bt_node.get_logger().error(f"트리 셋업 실패: {e}")
        return

    bt_node.get_logger().info("데이터 파이프라인 테스트를 10Hz로 시작합니다.")
    
    # 7. 실행 루프 (100ms 간격)
    try:
        tree.tick_tock(period_ms=100)
        rclpy.spin(bt_node)
    except KeyboardInterrupt:
        tree.interrupt()
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
    

