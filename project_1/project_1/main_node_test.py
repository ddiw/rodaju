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
from recycle_interfaces.action import PickPlace




rclpy.init()



class  Modes(Enum):
    STANDBY    = 0
    READY      = 1
    SORTTING   = 2
    PUASE      = 3
    ERROR      = 4
    DETECT     = 5
    IDLE       = 100
    TEST       = 500


class ManageNode(Node):
    def __init__(self):
        super().__init__("main_node")
        self.TestMode = False
        self.mode = Modes.STANDBY
        
        self._action_client = ActionClient(self,PickPlace,"/recycle/exec/pick_place")
        self.is_robot_moving = False

        self.create_subscription(Detections2D,"/recycle/vision/detections",self.detect_callback,10)
        self.target_list = []
        self.priority = ['paper_cup','can','pet']
        self.exclude = ['can']
        
        #시뮬레이션용 테스트 코드 
        self.done_target_list = []


    #비전에서 받은 데이터를 우선순위에 맟쳐 정렬하는 함수
    def detect_callback(self,msg):
        # 정렬 구현 성공!

        if self.is_robot_moving:
            return
        
        det = msg.detections 

        indexes = [x for x in self.priority if x not in self.exclude]
        valid_det = [i for i in det  if i.label in indexes]

        if not valid_det:
            self.get_logger().info("유효한 객체가 없습니ek!")
            return 
        
        self.target_list = sorted(valid_det, key = lambda x: (indexes.index(x.label), math.hypot(x.x_m, x.y_m)))
        self.sort_complete = True
        if self.TestMode == True: print(self.target_list)
        

        self.target = self.target_list[0]
        self.get_logger().info(f"최종 타겟 선정: {self.target.label} (거리: {math.hypot(self.target.x_m, self.target.y_m):.3f}m)")
    
        self.send_goal(self.target)
       
    
    def send_goal(self,target):
        self.get_logger().info(f"골좌표 도착! 라벨: {target.label} X = {target.x_m:.2f} ")
        self.get_logger().info("로봇 서버 대기중....----...---...--..-.")
        self._action_client.wait_for_server()
        self.is_robot_moving = True

        goal_msg = PickPlace.Goal()
        goal_msg.label = target.label
        goal_msg.has_3d = target.has_3d
        goal_msg.pick_x_m = target.x_m
        goal_msg.pick_y_m = target.y_m
        goal_msg.pick_z_m = target.z_m

        self.get_logger().info("목표값 전송!")

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
            self.done_target_list.append()
        else :
            self.get_logger().info("작업실패 ㅠㅠ")
        self.is_robot_moving = False

    def standby(self):
        if self.mode == Modes.STANDBY:
            self.get_logger().info("standby")



def main(args=None):
    node = ManageNode()
    # example = [('plastic', [2, 4, 8]), ('plastic', [3, 2, 1]), ('plastic', [6, 7, 8]), ('plastic', [3, 9, 7]), ('plastic', [4, 9, 1]), ('plastic', [2, 4, 8]), ('plastic', [3, 2, 1]), ('plastic', [6, 7, 8]), ('plastic', [3, 9, 7]), ('plastic', [4, 9, 1]), ('plastic', [2, 4, 8]), ('plastic', [3, 2, 1]), ('plastic', [6, 7, 8]), ('plastic', [3, 9, 7]), ('plastic', [4, 9, 1]), ('plastic', [2, 4, 8]), ('plastic', [3, 2, 1]), ('plastic', [6, 7, 8]), ('plastic', [3, 9, 7]), ('plastic', [4, 9, 1]), ('plastic', [2, 4, 8]), ('plastic', [3, 2, 1]), ('plastic', [6, 7, 8]), ('plastic', [3, 9, 7]), ('plastic', [4, 9, 1]), ('plastic', [2, 4, 8]), ('plastic', [3, 2, 1]), ('plastic', [6, 7, 8]), ('plastic', [3, 9, 7]), ('plastic', [4, 9, 1]), ('paper', [5, 3, 7]), ('paper', [1, 2, 5]), ('paper', [4, 9, 6]), ('paper', [7, 4, 2]), ('paper', [8, 9, 0]), ('paper', [5, 3, 7]), ('paper', [1, 2, 5]), ('paper', [4, 9, 6]), ('paper', [7, 4, 2]), ('paper', [8, 9, 0]), ('paper', [5, 3, 7]), ('paper', [1, 2, 5]), ('paper', [4, 9, 6]), ('paper', [7, 4, 2]), ('paper', [8, 9, 0]), ('paper', [5, 3, 7]), ('paper', [1, 2, 5]), ('paper', [4, 9, 6]), ('paper', [7, 4, 2]), ('paper', [8, 9, 0]), ('paper', [5, 3, 7]), ('paper', [1, 2, 5]), ('paper', [4, 9, 6]), ('paper', [7, 4, 2]), ('paper', [8, 9, 0]), ('paper', [5, 3, 7]), ('paper', [1, 2, 5]), ('paper', [4, 9, 6]), ('paper', [7, 4, 2]), ('paper', [8, 9, 0]), ('can', [1, 2, 3]), ('can', [5, 9, 5]), ('can', [5, 7, 1]), ('can', [2, 4, 1]), ('can', [1, 2, 3]), ('can', [5, 9, 5]), ('can', [5, 7, 1]), ('can', [2, 4, 1]), ('can', [1, 2, 3]), ('can', [5, 9, 5]), ('can', [5, 7, 1]), ('can', [2, 4, 1]), ('can', [1, 2, 3]), ('can', [5, 9, 5]), ('can', [5, 7, 1]), ('can', [2, 4, 1]), ('can', [1, 2, 3]), ('can', [5, 9, 5]), ('can', [5, 7, 1]), ('can', [2, 4, 1]), ('can', [1, 2, 3]), ('can', [5, 9, 5]), ('can', [5, 7, 1]), ('can', [2, 4, 1])]
    # example = [['plastic', [38, 24, 51]], ['can', [30, 17, 34]], ['plastic', [38, 5, 48]], ['paper', [11, 2, 24]], ['can', [34, 1, 30]], ['plastic', [24, 19, 43]], ['paper', [12, 17, 33]], ['can', [37, 22, 38]], ['paper', [7, 15, 54]], ['can', [11, 14, 26]], ['paper', [21, 17, 39]], ['plastic', [37, 29, 67]], ['paper', [14, 29, 54]], ['plastic', [25, 24, 31]], ['plastic', [10, 11, 46]], ['can', [7, 27, 37]], ['plastic', [12, 2, 16]], ['paper', [28, 15, 67]], ['can', [3, 25, 55]], ['plastic', [31, 12, 66]], ['paper', [10, 29, 51]], ['can', [7, 3, 12]], ['paper', [5, 8, 39]], ['can', [9, 12, 10]], ['paper', [9, 24, 47]], ['plastic', [24, 29, 55]], ['paper', [27, 25, 17]], ['plastic', [35, 20, 28]], ['plastic', [14, 22, 28]], ['can', [31, 26, 61]], ['plastic', [21, 29, 54]], ['paper', [20, 13, 47]], ['can', [3, 25, 62]], ['plastic', [35, 29, 13]], ['paper', [16, 4, 44]], ['can', [39, 8, 69]], ['paper', [7, 8, 37]], ['can', [10, 8, 57]], ['paper', [26, 25, 62]], ['plastic', [33, 25, 17]], ['paper', [3, 1, 42]], ['plastic', [31, 11, 58]], ['plastic', [35, 14, 43]], ['can', [15, 16, 68]], ['plastic', [25, 13, 58]], ['paper', [5, 18, 10]], ['can', [16, 8, 17]], ['plastic', [24, 19, 10]], ['paper', [8, 7, 29]], ['can', [31, 7, 68]], ['paper', [36, 29, 38]], ['can', [11, 19, 62]], ['paper', [26, 10, 25]], ['plastic', [16, 3, 14]], ['paper', [23, 26, 69]], ['plastic', [32, 14, 68]], ['plastic', [32, 9, 50]], ['can', [27, 7, 24]], ['plastic', [1, 13, 43]], ['paper', [37, 4, 45]], ['can', [22, 21, 36]], ['plastic', [24, 16, 19]], ['paper', [8, 22, 47]], ['can', [13, 19, 13]], ['paper', [22, 27, 55]], ['can', [3, 26, 27]], ['paper', [15, 25, 34]], ['plastic', [32, 9, 38]], ['paper', [20, 26, 45]], ['plastic', [15, 16, 47]], ['plastic', [35, 19, 10]], ['can', [13, 15, 53]], ['plastic', [13, 8, 19]], ['paper', [6, 20, 28]], ['can', [34, 14, 15]], ['plastic', [6, 16, 11]], ['paper', [31, 2, 12]], ['can', [12, 18, 27]], ['paper', [3, 29, 52]], ['can', [31, 5, 56]], ['paper', [11, 10, 23]], ['plastic', [25, 11, 14]], ['paper', [24, 8, 59]], ['plastic', [33, 13, 53]]]
    example = [('plastic', [33, 10, 30]), ('can', [12, 29, 30]), ('plastic', [3, 7, 58]), ('paper', [14, 3, 38]), ('can', [37, 21, 68]), ('plastic', [26, 15, 11]), ('paper', [15, 13, 54]), ('can', [38, 7, 19]), ('paper', [21, 18, 23]), ('can', [17, 20, 52]), ('paper', [28, 15, 22]), ('plastic', [23, 22, 49]), ('paper', [26, 12, 21]), ('plastic', [8, 11, 21]), ('plastic', [39, 19, 69]), ('can', [26, 6, 21]), ('plastic', [12, 9, 45]), ('paper', [5, 11, 26]), ('can', [14, 23, 46]), ('plastic', [38, 5, 16]), ('paper', [7, 21, 35]), ('can', [23, 22, 63]), ('paper', [23, 28, 49]), ('can', [31, 10, 48]), ('paper', [3, 28, 64]), ('plastic', [23, 26, 63]), ('paper', [8, 14, 60]), ('plastic', [35, 12, 42]), ('plastic', [30, 3, 64]), ('can', [22, 16, 13]), ('plastic', [28, 29, 13]), ('paper', [13, 19, 52]), ('can', [35, 1, 64]), ('plastic', [6, 26, 23]), ('paper', [28, 12, 50]), ('can', [13, 21, 52]), ('paper', [23, 6, 21]), ('can', [15, 28, 32]), ('paper', [36, 4, 41]), ('plastic', [32, 27, 53]), ('paper', [23, 17, 55]), ('plastic', [35, 4, 32]), ('plastic', [23, 19, 63]), ('can', [4, 19, 48]), ('plastic', [9, 15, 35]), ('paper', [30, 13, 69]), ('can', [23, 17, 23]), ('plastic', [31, 27, 34]), ('paper', [13, 13, 34]), ('can', [35, 29, 44]), ('paper', [28, 29, 24]), ('can', [10, 29, 44]), ('paper', [23, 12, 28]), ('plastic', [39, 12, 29]), ('paper', [5, 23, 50]), ('plastic', [32, 18, 36]), ('plastic', [9, 8, 37]), ('can', [30, 5, 32]), ('plastic', [16, 9, 51]), ('paper', [26, 10, 58]), ('can', [6, 28, 66]), ('plastic', [32, 3, 68]), ('paper', [1, 23, 47]), ('can', [39, 7, 15]), ('paper', [14, 2, 34]), ('can', [8, 26, 29]), ('paper', [29, 13, 41]), ('plastic', [3, 25, 26]), ('paper', [36, 18, 48]), ('plastic', [33, 24, 33]), ('plastic', [36, 2, 33]), ('can', [5, 26, 42]), ('plastic', [14, 22, 12]), ('paper', [35, 28, 60]), ('can', [22, 1, 55]), ('plastic', [29, 13, 42]), ('paper', [30, 7, 19]), ('can', [17, 19, 58]), ('paper', [17, 17, 59]), ('can', [7, 20, 23]), ('paper', [30, 9, 44]), ('plastic', [3, 2, 59]), ('paper', [39, 20, 27]), ('plastic', [30, 10, 28])]

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()