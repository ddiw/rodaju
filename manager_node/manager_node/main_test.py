import math
import py_trees
import py_trees_ros
from py_trees.common import Status, Access
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from recycle_interfaces.msg import SortCommand, Detections2D
from recycle_interfaces.action import MoveRobots
# from recycle_interfaces.action import GripperAction # 실제 그리퍼 액션 메시지에 맞게 임포트 필요


# ---------------------------------------------------------
# 1. 시스템 제어 및 흐름 통제 노드
# ---------------------------------------------------------
class Command_Control(py_trees.behaviour.Behaviour):
    """0순위: 커맨드 판단, 트리 흐름 제어 및 초기 데이터 장전"""
    def __init__(self, name="Command_Control"):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(key="cmd_data", access=Access.READ)
        self.blackboard.register_key(key="works", access=Access.WRITE)
        self.blackboard.register_key(key="phase", access=Access.WRITE)
        
        # 고정 좌표 장전을 위한 쓰기 권한
        self.blackboard.register_key(key="det_pos", access=Access.WRITE)
        self.blackboard.register_key(key="bin_pos", access=Access.WRITE)
        self.is_paused = False

    def update(self):
        if self.blackboard.exists("cmd_data") and self.blackboard.cmd_data.cmd:
            cmd = self.blackboard.cmd_data.cmd
            self.blackboard.cmd_data.cmd = "" 
            
            if cmd == "STOP":
                self.logger.error("🚨 비상정지(STOP): 작업을 종료하고 트리를 차단합니다.")
                self.blackboard.works = False
                self.is_paused = False
                return Status.SUCCESS 
                
            elif cmd == "PAUSE":
                self.logger.warn("⏸️ 일시정지(PAUSE): 트리를 차단합니다.")
                self.is_paused = True
                return Status.SUCCESS 
                
            elif cmd == "RESUME":
                self.logger.info("▶️ 재시작(RESUME): 차단을 해제합니다.")
                self.is_paused = False
                
            elif cmd == "START":
                self.logger.info("▶️ 시작(START): 시스템을 초기화하고 고정 좌표를 장전합니다.")
                self.blackboard.works = True
                self.blackboard.phase = 1
                self.is_paused = False

                # [필수 방어 로직] Phase 1, Phase 7용 고정 좌표 사전 장전
                det_goal = MoveRobots.Goal()
                det_goal.move_mode = "movej"
                det_goal.goal_pos = [-11.0, 26.0, 19.0, 0.0, 133.0, -12.0]
                det_goal.vel, det_goal.acc = 20.0, 20.0
                self.blackboard.det_pos = det_goal

                bin_goal = MoveRobots.Goal()
                bin_goal.move_mode = "movej"
                bin_goal.goal_pos = [350.0, -470.0, 400.0, 0.0, -180.0, 0.0] # 실제 쓰레기통 좌표로 수정 필요
                bin_goal.vel, bin_goal.acc = 20.0, 20.0
                self.blackboard.bin_pos = bin_goal

        if self.is_paused:
            return Status.SUCCESS 
            
        return Status.FAILURE


class Check_Works(py_trees.behaviour.Behaviour):
    """1순위 진입점: 가동 상태(works) 확인"""
    def __init__(self, name="Check_Works"):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(key="works", access=Access.READ)

    def update(self):
        if self.blackboard.exists("works") and self.blackboard.works:
            return Status.SUCCESS
        return Status.FAILURE


# ---------------------------------------------------------
# 2. 핵심 연산 및 스마트 액션 노드
# ---------------------------------------------------------
class Calculate_target(py_trees.behaviour.Behaviour):
    """Phase 2: 타겟 연산 및 Phase 3, 4, 6 궤적 사전 장전"""
    def __init__(self, name="Calculate_target"):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(key="detect_datas", access=Access.READ)
        self.blackboard.register_key(key="cmd_data", access=Access.READ)
        self.blackboard.register_key(key="works", access=Access.WRITE)
        self.blackboard.register_key(key="phase", access=Access.READ)
        self.blackboard.register_key(key="phase", access=Access.WRITE)
        
        self.blackboard.register_key(key="approach_pos", access=Access.WRITE)
        self.blackboard.register_key(key="pick_pos", access=Access.WRITE)
        self.blackboard.register_key(key="lift_pos", access=Access.WRITE)

    def update(self):
        # 이미 2단계를 지났다면 건너뜀
        if self.blackboard.exists("phase") and self.blackboard.phase > 2:
            return Status.SUCCESS

        det_msg = self.blackboard.detect_datas
        if not det_msg.detections:
            self.logger.info("비전 데이터에 객체가 없습니다. 작업 종료.")
            self.blackboard.works = False
            return Status.FAILURE

        priority = self.blackboard.cmd_data.priority_order 
        exclude = self.blackboard.cmd_data.exclude_mask
        cls = [i for i in priority if i not in exclude]

        valid_det = [i for i in det_msg.detections if i.label in cls]
        if not valid_det: 
            self.logger.info("우선순위 타겟이 없습니다. 작업 종료.")
            self.blackboard.works = False
            return Status.FAILURE

        targets_sort = sorted(valid_det, key=lambda x: (cls.index(x.label), math.hypot(x.x_m, x.y_m)))
        target_x, target_y, target_z, target_angle = targets_sort[0].x_m, targets_sort[0].y_m, targets_sort[0].z_m, targets_sort[0].angle_deg
     
        # 궤적 사전 연산
        goal_approach = MoveRobots.Goal()
        goal_approach.move_mode, goal_approach.goal_pos = "det_move", [target_x, target_y, target_z + 80.0, target_angle]
        goal_approach.vel, goal_approach.acc = 20.0, 20.0
        self.blackboard.approach_pos = goal_approach

        goal_pick = MoveRobots.Goal()
        goal_pick.move_mode, goal_pick.goal_pos = "det_move", [target_x, target_y, target_z, target_angle]
        goal_pick.vel, goal_pick.acc = 10.0, 10.0
        self.blackboard.pick_pos = goal_pick

        goal_lift = MoveRobots.Goal()
        goal_lift.move_mode, goal_lift.goal_pos = "det_move", [target_x, target_y, target_z + 150.0, target_angle]
        goal_lift.vel, goal_lift.acc = 20.0, 20.0
        self.blackboard.lift_pos = goal_lift

        self.logger.info(f"타겟 지정 완료: {targets_sort[0].label}. 궤적 장전 완료.")
        
        # 사후 갱신
        self.blackboard.phase = 3
        return Status.SUCCESS


class Smart_Phase_Action(py_trees_ros.action_clients.FromBlackboard):
    """사전 검증과 사후 갱신을 캡슐화한 지능형 액션 클라이언트"""
    def __init__(self, name, phase_num, action_type, action_name, key):
        super().__init__(name=name, action_type=action_type, action_name=action_name, key=key)
        self.phase_num = phase_num
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(key="phase", access=Access.READ)
        self.blackboard.register_key(key="phase", access=Access.WRITE)

    def update(self):
        if self.blackboard.exists("phase") and self.blackboard.phase > self.phase_num:
            self.logger.info(f"⏩ Phase {self.phase_num} 이미 완료됨. 생략합니다.")
            return Status.SUCCESS

        status = super().update()

        if status == Status.SUCCESS:
            next_phase = self.phase_num + 1 if self.phase_num < 8 else 1
            self.blackboard.phase = next_phase
            self.logger.info(f"✅ Phase {self.phase_num} 완료. Phase {next_phase}(으)로 갱신.")

        return status


class Home_pos(py_trees.behaviour.Behaviour):
    """작업 종료 시 홈 복귀 좌표 셋업"""
    def __init__(self, name="Home_pos"):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(key="home_pos", access=Access.WRITE)
    
    def update(self):
        home_pos = MoveRobots.Goal()
        home_pos.move_mode = "movej"
        home_pos.goal_pos = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
        home_pos.vel, home_pos.acc = 30.0, 30.0
        self.blackboard.home_pos = home_pos
        return Status.SUCCESS


# ---------------------------------------------------------
# 3. 메인 어셈블리
# ---------------------------------------------------------
def main():
    rclpy.init()
    bt_node = rclpy.create_node("main_node")

    # 1. 스마트 액션 노드 인스턴스 생성
    action_phase1 = Smart_Phase_Action(name="Phase1_Detect",   phase_num=1, action_type=MoveRobots, action_name="/recycle/exec/move_robot", key="det_pos")
    calculate_target = Calculate_target(name="Phase2_Calculate")
    
    action_phase3 = Smart_Phase_Action(name="Phase3_Approach", phase_num=3, action_type=MoveRobots, action_name="/recycle/exec/move_robot", key="approach_pos")
    action_phase4 = Smart_Phase_Action(name="Phase4_Close",    phase_num=4, action_type=MoveRobots, action_name="/recycle/exec/move_robot", key="pick_pos")
    action_phase6 = Smart_Phase_Action(name="Phase6_Lift",     phase_num=6, action_type=MoveRobots, action_name="/recycle/exec/move_robot", key="lift_pos")
    action_phase7 = Smart_Phase_Action(name="Phase7_Bin",      phase_num=7, action_type=MoveRobots, action_name="/recycle/exec/move_robot", key="bin_pos")

    # 그리퍼 액션 클라이언트 (인터페이스 맞춤 후 주석 해제)
    # action_phase5 = Smart_Phase_Action(name="Phase5_Grip",    phase_num=5, action_type=GripperAction, action_name="/recycle/exec/gripper", key="grip_cmd")
    # action_phase8 = Smart_Phase_Action(name="Phase8_Release", phase_num=8, action_type=GripperAction, action_name="/recycle/exec/gripper", key="release_cmd")

    # 2. 메인 시퀀스 (Pick_Sequence) 조립
    check_works = Check_Works(name="Check_Works")
    execution_sequence = py_trees.composites.Sequence(name="Execution_Sequence", memory=True)
    execution_sequence.add_children([
        action_phase3, 
        action_phase4,
        # action_phase5, 
        action_phase6,
        action_phase7,
        # action_phase8
    ])

    pick_sequence = py_trees.composites.Sequence(name="Pick_Sequence", memory=True)
    pick_sequence.add_children([
        check_works,
        action_phase1,
        calculate_target,
        execution_sequence
    ])

    # 3. 종료 시퀀스 (Done_Sequence) 조립
    home_pos = Home_pos(name="Home_pos")
    home_move_action = py_trees_ros.action_clients.FromBlackboard(name="home_move_action", action_type=MoveRobots, action_name="/recycle/exec/move_robot", key="home_pos")
    idle_standby = py_trees.behaviours.Running(name="Idle_Standby")
    
    done_sequence = py_trees.composites.Sequence(name="Done_Sequence", memory=True)
    done_sequence.add_children([home_pos, home_move_action, idle_standby])

    # 4. 최상위 미션 컨트롤 조립
    cmd_control = Command_Control(name="Command_Control")
    mission_control = py_trees.composites.Selector(name="Mission_Control", memory=False)
    mission_control.add_children([
        cmd_control,    # 0순위 (통제기)
        pick_sequence,  # 1순위 (메인 루틴)
        done_sequence   # 2순위 (종료 루틴)
    ])

    # 5. 토픽 구독자 연결
    default_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE)
    cmd_to_black_board = py_trees_ros.subscribers.ToBlackboard(name="cmd_to_black_board", topic_name="/recycle/command", qos_profile=default_qos, topic_type=SortCommand, blackboard_variables={"cmd_data": None}, clearing_policy=py_trees.common.ClearingPolicy.NEVER)
    detect_to_black_board = py_trees_ros.subscribers.ToBlackboard(name="detect_to_black_board", topic_name="/recycle/vision/detections", qos_profile=default_qos, topic_type=Detections2D, blackboard_variables={"detect_datas": None}, clearing_policy=py_trees.common.ClearingPolicy.NEVER)
    
    sub_parallel = py_trees.composites.Parallel(name="sub_Parallel", policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False))
    sub_parallel.add_children([cmd_to_black_board, detect_to_black_board])

    root_parallel = py_trees.composites.Parallel(name="Root_Parallel", policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False))
    root_parallel.add_children([mission_control, sub_parallel])

    # 트리 가동
    tree = py_trees_ros.trees.BehaviourTree(root_parallel)
    try:
        tree.setup(node=bt_node)
    except Exception as e:
        bt_node.get_logger().error(f"Tree setup failed: {e}")
        return

    bt_node.get_logger().info("✅ 계층형 BT 데이터 파이프라인 가동 완료 (최적화 버전)")
    
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