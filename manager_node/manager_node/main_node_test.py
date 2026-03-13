import os
import time
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from enum import Enum
from recycle_interfaces.msg import  Detections2D
from collections import deque 
import math

from rclpy.action import ActionClient
from recycle_interfaces.action import PickPlace,Detectinginit
from recycle_interfaces.msg import SortCommand
import time

class  Modes(Enum):
    STANDBY    = 0
    READY      = 1
    DETECTING  = 2
    SORTTING   = 3
    PUASE      = 4
    ERROR      = 5
    IDLE       = 100
    TEST       = 500


class ManageNode(Node):
    def __init__(self):
        super().__init__("main_node")
        self.TestMode = False
        self.mode = Modes.STANDBY
        self.mode_memory = ""
      
        self._action_client = ActionClient(self,PickPlace,"/recycle/exec/pick_place")
        #self._detect_action_client = ActionClient(self,Detecting_init,"recycle/exec/detect_init")

        self.is_robot_moving = False
        
        
        self.create_subscription(SortCommand,"/recycle/command",self.cmd_callback,10)
        self.create_subscription(Detections2D,"/recycle/vision/detections",self.detect_callback,10)
        self.target_list = []
        self.priority = ['paper','can','plastic']
        self.exclude = []

        self.stay_detect = 0


    def cmd_callback(self,msg):
        command = msg.cmd
        priority = msg.priority_order
        exclude = msg.exclude_mask
        raw_text = msg.raw_text
        mod = [Modes.SORTTING,Modes.DETECTING]
        #받은 커맨드 데이터 + 현재 모드 출력 
        self.get_logger().info(f"{command} + {self.mode} + {priority} + {exclude} + {raw_text}")

        #대기상태일때 시작하는 커맨드
        if command == 'START': 
            if self.mode == Modes.STANDBY: 
                self.send_goal()
                self.mode = Modes.READY
            self.get_logger().info(f"{self.mode}")
            return 

        #동작중 비상정지 커맨드 
        elif command == 'PAUSE': 
            if self.mode in mod:
                self.mode_memory = self.mode
                self.mode = Modes.PUASE
            self.get_logger().info(f"mode: {self.mode} / mode_memory: {self.mode_memory}")
            return 
        
        #비상정지중 재시작 커맨드
        elif command == 'RESUME':
            #저장된 이전모드로 다시시작
            if self.mode == Modes.PUASE:    
                self.mode = self.mode_memory
                self.mode_memory = Modes.STANDBY
                
            self.get_logger().info(f"resume_mode: {self.mode}")
            return 
            
        #우선순위 변경 커맨드 동작중이라면 다음 동작부터 적용 
        if command == "SET_POLICY":
            self.get_logger().info(f"priority: {priority}  exclude: {exclude}")
            filltered_priority= [i  for i in self.priority if i not in priority]
            self.priority = priority + filltered_priority
            self.exclude = exclude





    """ 비전에서 받은 데이터를 우선순위에 맟쳐 정렬하는 함수 """
    def detect_callback(self,msg):
        
        
        if self.mode != Modes.DETECTING :
            return 
       
        det = msg.detections 
        self.get_logger().info(f"{det}") #감지된 객체 출력 

        #분류 제외할 객체 이름 삭제 
        indexes = [x for x in self.priority if x not in self.exclude]
        
        #디텍팅된 객체 저장할 리스트 
        valid_det = []
        for i in det:
            if i.label in indexes:
                valid_det.append(i)

        #5번 토픽을 받을동안 디텍팅된 객체가 없다면 모드 변경 
        if not valid_det:
            self.stay_detect += 1
            if self.stay_detect == 5:
                self.get_logger().info("유효한 객체가 없습니ek!")
                self.mode = Modes.STANDBY
                self.stay_detect = 0
            return 
        else: self.stay = 0
        
        #정렬하는 코드 sorted()함수 : ~~~~
        self.target_list = sorted(valid_det, key = lambda x: (indexes.index(x.label), math.hypot(x.x_m, x.y_m)))
        self.sort_complete = True
        if self.TestMode == True: print(self.target_list)
        

        self.target = self.target_list[0]
        self.get_logger().info(f"최종 타겟 선정: {self.target.label} (거리: {math.hypot(self.target.x_m, self.target.y_m):.3f}m)")
        
        self.mode = Modes.SORTTING
        self.send_goal(self.target)
        
    


    """ 액션 함수들 """

    def det_init(self):
        self.get_logger().info("디텍팅 위치로 이동!")
        self._detect_action_client.wait_for_server()
        self.is_robot_moving = True
        send_det_future = self._detect_action_client.send_goal_async(True,feedback_callback=self.det_f_cb)
        send_det_future.add_done_callback(self.det_response_cb)

    def det_f_cb(self,feedback_msg):
        return 

    def det_response_cb(self,future):
        result = future.result()
        if not result.accepted:
            self.get_logger().error("액션 서버에서 거절했습니다.(로봇 동작중)")
            self.is_robot_moving = False
            return 
        self.get_logger().info("명령 수락! 로봇 동작!")

        get_result_future = result.get_result_async()
        get_result_future.add_done_callback(self.det_result_cb)


    def det_result_cb(self,future):
        result = future.result().result
        if result.success:
            self.get_logger().info("디텍팅 위치 이동완료")
        else : self.get_logger().info("위치 이동 실패")
        self.is_robot_moving = False
        self.mode = Modes.DETECTING
    



    """ 디텍팅된 객체로 이동! """
    def send_goal(self,target):
        self.get_logger().info(f"골좌표 도착! 라벨: {target.label} X = {target.x_m:.2f} ")
        self.get_logger().info("로봇 서버 대기중....----...---...--..-.")
        self._action_client.wait_for_server()
        self.is_robot_moving = True
        goal_msg = PickPlace.Goal()
        goal_msg.label = target.label
        goal_msg.has_3d = target.has_3d
        goal_msg.pick_x_m = target.x_m
        goal_msg.pick_y_m = target.y_m + 0.02
        goal_msg.pick_z_m = (target.z_m + 0.03)
        goal_msg.bin_id = f"BIN_{target.label}"

        self.get_logger().info(f"목표값 전송! {goal_msg.pick_z_m:.3f}")

        send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback) 
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self,future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("액션 서버에서 거절했습니다.(로봇 동작중)")
            self.is_robot_moving = False
            return 
        
        self.get_logger().info("명령 수락! 로봇 동작!")

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self,feedback_msg):
        progress = feedback_msg.feedback.progress
        phase = feedback_msg.feedback.phase
        self.get_logger().info(f"로봇 진행상황: {phase} {progress:.0f}%")

    def get_result_callback(self,future):
        result = future.result().result

        if result.success:
            self.get_logger().info("작업 성공!")
        else :
            self.get_logger().info("작업실패 ㅠㅠ")
        self.is_robot_moving = False
        self.mode = Modes.DETECTING

        



def main(args=None):
    rclpy.init(args=args)
    node = ManageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()