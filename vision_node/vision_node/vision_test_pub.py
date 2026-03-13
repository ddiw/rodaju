import os
import time
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from enum import Enum
from recycle_interfaces.msg import  Detections2D, Detection2D
from collections import deque 



labels = ['pet','can','paper_cup','can','pet','can','paper_cup','pet']
confidence = [0.94,0.91,0.91,0.90,0.88,0.87,0.78,0.76]
x = [289,131,-50,515,280,476,217,405]
y = [-65,293,74,407,213,221,296,337]
w = [312,160,212,120,235,166,215,117]
h = [248,163,245,75,116,169,222,152]
cx = [445,211,56,575,397,559,324,463]
cy = [59,374,196,444,271,305,407,413]
x_m = [0.065,-0.069,-0.147,0.139,0.039,0.131,-0.002,0.086]
y_m = [-0.103,0.077,-0.026,0.112,0.015,0.034,0.104,0.108]
z_m = [0.337,0.361,0.328,0.34,0.347,0.345,0.388,0.389]
has_3d = [True,True,True,True,True,True,True,True]

class V_test_pub(Node):
    def __init__(self):
        super().__init__("test_node1")

        self.pub = self.create_publisher(Detections2D,"/recycle/vision/detections",10)
      


        self.timer = self.create_timer(5.0, self.time_callback)


    def time_callback(self):
        dets = []
        for i in range(0,8):
            dets.append(self.detection_data(i))

        msg = Detections2D()
        msg.frame_id = "camera_color_optical_frame"
        msg.detections = dets

        self.pub.publish(msg)
        self.get_logger().info("publish!!")

    def detection_data(self,i):
        det = Detection2D()
        det.id = i
        det.label      = labels[i]
        det.confidence = confidence[i]
        det.x,det.y    = x[i] , y[i]
        det.w,det.h    = w[i] , h[i]
        det.cx,det.cy  = cx[i] , cy[i]
        det.x_m, det.y_m, det.z_m = x_m[i], y_m[i], z_m[i]
        det.has_3d = True
        return det 



def main(args=None):
    rclpy.init(args=args)
    vt = V_test_pub()
    try:
        rclpy.spin(vt)
    except KeyboardInterrupt:
        pass
    finally:
        vt.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    main()