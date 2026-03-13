import math
import py_trees
import py_trees_ros
from py_trees.common import Status, Access
import rclpy
from recycle_interfaces.msg import SortCommand, Detections2D
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from recycle_interfaces.action import MoveRobots,ControlGripper
import time

class Commands(py_trees.behaviour.Behaviour):
    def __init__(self, name = "Commands"):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name="Global")
        self.blackboard.register_key(key="cmd_data",access=Access.READ)
        self.blackboard.register_key(key="det_pos",access=Access.WRITE)
        self.blackboard.register_key(key="works",access=Access.WRITE)
        self.blackboard.register_key(key="phase",access=Access.WRITE)


    def update(self):
        if self.blackboard.exists("cmd_data"):
            if  self.blackboard.cmd_data.cmd == "STOP":
                self.logger.info("비상정지 커맨드!! 프로세스 비상정지 합니다")
                self.blackboard.works = False
                self.blackboard.cmd_data.cmd = ""
                return py_trees.common.Status.SUCCESS
            
            elif self.blackboard.cmd_data.cmd == "START":
                self.logger.info("시작 커맨드!! 프로세스 시작 합니다")
                self.blackboard.works = True

                if not self.blackboard.exists("det_pos"):
                    goal_msg = MoveRobots.Goal()
                    goal_msg.move_mode = "movej"
                    goal_msg.goal_pos = [-11.0, 26.0, 19.0, 0.0, 133.0, -12.0]
                    goal_msg.goal_pos_c = []
                    goal_msg.vel = 50
                    goal_msg.acc = 50
                    self.blackboard.det_pos = goal_msg

                if not self.blackboard.exists("phase"): self.blackboard.phase = 1
                self.blackboard.cmd_data.cmd = ""
                return py_trees.common.Status.FAILURE
        
        if self.blackboard.exists("works") and self.blackboard.works:
            return Status.FAILURE

        return py_trees.common.Status.FAILURE
    


    
# phase 2
class Calculate_target(py_trees.behaviour.Behaviour):
    def __init__(self, name = "Caculate_target"):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name="Global")
        self.blackboard.register_key(key="detect_datas",access=Access.READ)
        self.blackboard.register_key(key="cmd_data",access=Access.READ)
        self.blackboard.register_key(key="move_pos",access=Access.WRITE)
        self.blackboard.register_key(key="works",access=Access.WRITE)
        self.blackboard.register_key(key="phase",access=Access.WRITE)
        self.blackboard.register_key(key="approach_pos",access=Access.WRITE)
        self.blackboard.register_key(key="close_pos",access=Access.WRITE)
        self.blackboard.register_key(key="lift_pos",access=Access.WRITE)
        self.blackboard.register_key(key="home_pos",access=Access.WRITE)
        self.blackboard.register_key(key="bin_pos",access=Access.WRITE)
        self.blackboard.register_key(key="done",access=Access.WRITE)
        self.blackboard.register_key(key="gripper_open", access=Access.WRITE)
        self.blackboard.register_key(key="gripper_close", access=Access.WRITE)

        self.wait_duration = 2.0  # 대기 시간 설정 (초)
        self.wait_start_time = 0.0
        self.is_waiting = False

    def initialise(self):
        # 노드가 처음 틱을 받거나 재시작될 때 초기화
        self.is_waiting = False
        super().initialise()

    def terminate(self, new_status):
        # 중단(STOP)되거나 완료되었을 때 상태 초기화
        self.is_waiting = False
        super().terminate(new_status)

        
    def update(self):
        
        if self.blackboard.phase > 2: return Status.SUCCESS
        elif self.blackboard.phase < 2: return Status.FAILURE

        # 1. 의도적 지연(Delay) 로직 수행
        if not self.is_waiting:
            self.logger.info(f"디텍트 위치 도달. 비전 데이터 안정화를 위해 {self.wait_duration}초 대기합니다.")
            self.wait_start_time = time.time()
            self.is_waiting = True
            return Status.RUNNING
        
        # 지정된 시간이 지나지 않았다면 계속 RUNNING 상태 반환 (트리 블로킹 방지)
        if time.time() - self.wait_start_time < self.wait_duration:
            return Status.RUNNING
        
        if self.blackboard.phase > 2:return Status.SUCCESS
        elif self.blackboard.phase < 2 : return Status.FAILURE

        if not self.blackboard.exists("detect_datas"):
            return Status.FAILURE

        det_msg = self.blackboard.detect_datas
        if not det_msg.detections:
            self.logger.info("비전 데이터에 객체가 전혀 없습니다. FAILURE 반환.")
            self.blackboard.works = False
            self.blackboard.phase = 0 
            self.blackboard.done = True 


            return py_trees.common.Status.FAILURE

        priority = self.blackboard.cmd_data.priority_order 
        exclude = self.blackboard.cmd_data.exclude_mask

        cls = [i for i in priority if i not in exclude]

        valid_det = [i for i in det_msg.detections if i.label in cls]
        if not valid_det : 
            self.blackboard.works = False
            self.blackboard.phase = 0 
            self.blackboard.done = True 
            self.logger.info("조건에 맞는 객체가 없어서 작업 종료")
            return Status.FAILURE

        
        targets_sort = sorted(valid_det, key = lambda x: (cls.index(x.label), math.hypot(x.x_m, x.y_m)))
     
        target = MoveRobots.Goal()
        target.move_mode = "apporach"
        target.goal_pos = [targets_sort[0].x_m, targets_sort[0].y_m, targets_sort[0].z_m, targets_sort[0].angle_deg]
        target.vel = 50
        target.acc = 50
        self.blackboard.approach_pos = target 

        pick_pos = MoveRobots.Goal()
        pick_pos.move_mode = "pick_pos"
        pick_pos.goal_pos = [targets_sort[0].x_m, targets_sort[0].y_m, targets_sort[0].z_m, targets_sort[0].angle_deg]
        pick_pos.vel = 50
        pick_pos.acc = 50
        self.blackboard.close_pos = pick_pos


        lift_pos = MoveRobots.Goal()
        lift_pos.move_mode = "lift"
        lift_pos.goal_pos = [targets_sort[0].x_m, targets_sort[0].y_m, targets_sort[0].z_m, targets_sort[0].angle_deg]
        lift_pos.vel = 50
        lift_pos.acc = 50
        self.blackboard.lift_pos = lift_pos

        home_pos = MoveRobots.Goal()
        home_pos.move_mode = "movej"
        home_pos.goal_pos = [0.0,0.0,90.0,0.0,90.0,0.0]
        home_pos.vel = 70
        home_pos.acc = 7
        self.blackboard.home_pos = home_pos


        bin_pos = MoveRobots.Goal()
        bin_pos.move_mode = "movel"
        bin_pos.vel = 70
        bin_pos.acc = 70
        if targets_sort[0].label == "plastic":
            bin_pos.goal_pos = [446.0, -520.0, 260.0, 90.0, -150.0, 90.0]
        if targets_sort[0].label == "can":
            bin_pos.goal_pos =  [280.0, -520.0, 260.0, 90.0, -150.0, 90.0]
        if targets_sort[0].label == "paper":
            bin_pos.goal_pos =[120.0, -520.0, 260.0, 90.0, -150.0, 90.0]
        self.blackboard.bin_pos = bin_pos


        grip_close = ControlGripper.goal()
        grip_close.command = "close"
        if targets_sort[0].label == "plastic":
            grip_close.force = 300
        if targets_sort[0].label == "can":
            grip_close.force = 350
        if targets_sort[0].label == "paper":
            grip_close.force = 200

        self.blackboard.gripper_close = grip_close

        grip_open = ControlGripper.goal()
        grip_open.command = "open"

        self.blackboard.gripper_open = grip_open

     
        self.logger.info(f"타겟지정 완료  종류: {targets_sort[0].label}")
        self.blackboard.phase = 3
        self.logger.info("페이즈2 완료")
        return Status.SUCCESS


#
class Check_Works(py_trees.behaviour.Behaviour):
    """1순위 진입점: 작업 상태일 때만 메인 시퀀스 실행 허가"""
    def __init__(self, name="Check_Works"):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name="Global")
        self.blackboard.register_key(key="works", access=Access.READ)
        self.blackboard.register_key(key="done",access=Access.READ)

    def update(self):
        if self.blackboard.exists("works") and self.blackboard.works:
            return Status.SUCCESS
        
        
        return Status.FAILURE

#
class Smart_Phase_Action(py_trees_ros.action_clients.FromBlackboard):
    """
    initialise() 생명주기까지 완벽히 통제하여 Goal 접근 경고를 원천 차단한 
    최종 진화 형태의 지능형 단일 액션 노드.
    """
    def __init__(self, name, phase_num, action_type, action_name, key):
        super().__init__(
            name=name, 
            action_type=action_type, 
            action_name=action_name, 
            key=key 
        )
        self.phase_num = phase_num
        self.global_board = py_trees.blackboard.Client(name="Global")
        self.global_board.register_key(key="phase", access=py_trees.common.Access.READ)
        self.global_board.register_key(key="phase", access=py_trees.common.Access.WRITE)
        self.global_board.register_key(key=key, access=py_trees.common.Access.READ)
    def initialise(self):
        if self.global_board.exists("phase") and self.global_board.phase > self.phase_num:
            self.logger.info(f"Phase {self.phase_num} 이미 완료됨. Goal 데이터 접근을 차단합니다.")
            return
        
        super().initialise()

    def update(self):
        if self.global_board.exists("phase") and self.global_board.phase > self.phase_num:
            return py_trees.common.Status.SUCCESS

        status = super().update()

        if status == py_trees.common.Status.SUCCESS:
            next_phase = self.phase_num + 1 if self.phase_num < 7 else 1
            self.global_board.phase = next_phase
            self.logger.info(f"Phase {self.phase_num} 완료. Phase {next_phase}(으)로 갱신.")

        return status



class Home_pos(py_trees.behaviour.Behaviour):
    def __init__(self, name="Home_pos"):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name="Global")
        self.blackboard.register_key(key="home_pos",access=Access.WRITE)
        self.blackboard.register_key(key="done",access=Access.READ)
        self.blackboard.register_key(key="done",access=Access.WRITE)
        self.blackboard.register_key(key="works",access=Access.READ)
    def update(self):

        if self.blackboard.exists("done") and self.blackboard.done == True and not self.blackboard.works :
            self.blackboard.done = None
            return Status.SUCCESS
        return Status.FAILURE



def main():
    rclpy.init()
    bt_node = rclpy.create_node("main_node")


    
    det_move_action = py_trees_ros.action_clients.FromBlackboard(name="det_move_action",action_type=MoveRobots,action_name="/recycle/exec/move_robot",key="det_pos")
    # close_move_action = py_trees_ros.action_clients.FromBlackboard(name="close_move_action",action_type=MoveRobot,action_name="/recycle/exec/move_robot",key="move_pos")
    
    check_works = Check_Works(name="Check_Works")    
    action_phase1_det = Smart_Phase_Action(name="Phase1_detect", phase_num=1, action_type=MoveRobots, action_name="/recycle/exec/move_robot", key="det_pos")
    calculate_target = Calculate_target(name="Caculate_target")
    action_phase3_approach = Smart_Phase_Action(name="Phase3_approach", phase_num=3, action_type=MoveRobots, action_name="/recycle/exec/move_robot", key="approach_pos")
    action_phase4_close = Smart_Phase_Action(name="Phase4_close", phase_num=4, action_type=MoveRobots, action_name="/recycle/exec/move_robot",key="close_pos")
    action_phase5_grip = Smart_Phase_Action(name="Phase5_grip",phase_num=5, action_type=ControlGripper, action_name="/recycle/exec/control_gripper",key="grip_close")
    action_phase6_lift = Smart_Phase_Action(name="Phase6_lift", phase_num=6, action_type=MoveRobots, action_name="/recycle/exec/move_robot",key="lift_pos")
    action_phase7_home = Smart_Phase_Action(name="Phase7_home", phase_num=7, action_type=MoveRobots, action_name="/recycle/exec/move_robot",key="home_pos")
    action_phase8_bin = Smart_Phase_Action(name="Phase8_bin",phase_num=8, action_type=MoveRobots, action_name="/recycle/exec/move_robot",key="bin_pos")
    action_phase9_place = Smart_Phase_Action(name="Phase9_place",phase_num=9, action_type=ControlGripper, action_name="/recycle/exec/control_gripper",key="grip_open")
    

    pick_sequence = py_trees.composites.Sequence(name="Pick_Sequence", memory=True)
    pick_sequence.add_children([
        check_works,
        action_phase1_det,
        calculate_target, 
        action_phase3_approach ,
        action_phase4_close,
        action_phase5_grip,
        action_phase6_lift,
        action_phase7_home,
        action_phase8_bin,
        action_phase9_place,
    ])

    #홈 위치 복귀 
    home_pos = Home_pos(name="Home_pos")
    home_move_action = py_trees_ros.action_clients.FromBlackboard(name="home_move_action",action_type=MoveRobots,action_name="/recycle/exec/move_robot",key="home_pos")

    home_sequence = py_trees.composites.Sequence(name="Homesequence",memory=True)
    home_sequence.add_children([home_pos,home_move_action])

    task_fallback = py_trees.composites.Selector(name="Task_Fallback", memory=False)
    task_fallback.add_children([pick_sequence, home_sequence])


    #커맨드 판단 객체 
    cmd_stop_sequence = Commands("Commands")
    idle_running = py_trees.behaviours.Running(name="Idle_Standby")
    # 0순위: 비상 정지  # 1순위: 메인 픽업 임무   # 2순위: 대기 상태
    mission_control = py_trees.composites.Selector(name="Mission_Control", memory=False)
    mission_control.add_children([cmd_stop_sequence,task_fallback,idle_running])



    #subscriper
    #-----------------------------------------------------------------------
    default_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST,depth=10,reliability=ReliabilityPolicy.RELIABLE)
    
    cmd_to_black_board = py_trees_ros.subscribers.ToBlackboard(
        name="cmd_to_black_board",
        topic_name="/recycle/command",  
        qos_profile=default_qos,  
        topic_type=SortCommand,
        blackboard_variables={"cmd_data": None},
        clearing_policy=py_trees.common.ClearingPolicy.NEVER,  
    )

    detect_to_black_board = py_trees_ros.subscribers.ToBlackboard(
        name = "detect_to_black_board",
        topic_name = "/recycle/vision/detections",
        qos_profile = default_qos,
        topic_type=Detections2D,
        blackboard_variables={"detect_datas":None},
        clearing_policy=py_trees.common.ClearingPolicy.NEVER,
    )
    #-----------------------------------------------------------------------

    sub_parallel = py_trees.composites.Parallel(name="sub_Parallel",policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False))
    sub_parallel.add_children([cmd_to_black_board,detect_to_black_board])

    root_parallel = py_trees.composites.Parallel(name="Root_Parallel",policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False))
    root_parallel.add_children([mission_control,sub_parallel])


    #tree
    tree = py_trees_ros.trees.BehaviourTree(root_parallel)

    try:
        tree.setup(node=bt_node)
    except Exception as e:
        bt_node.get_logger().error(f"Tree setup failed: {e}")
        return

    bt_node.get_logger().info("계층형 BT 데이터 파이프라인 가동 완료!!")
    
    try:
        tree.tick_tock(period_ms=100)
        rclpy.spin(bt_node)
    except KeyboardInterrupt:
        tree.interrupt()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
if __name__=="__main__":
    main()