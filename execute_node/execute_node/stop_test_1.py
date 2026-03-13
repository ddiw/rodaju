import time
import rclpy
import DR_init

ROBOT_ID    = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id    = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# 목표 좌표
J_HOME = [0,0,90,0,90,0]
J_WORK = [-11.0, 26.0, 19.0, 0.0, 133.0, -12.0]


def main():
    rclpy.init()
    _dsr_node = rclpy.create_node("m0609_single_test_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = _dsr_node

    try:
        from DSR_ROBOT2 import amovej, posj
    except ImportError:
        _dsr_node.get_logger().error("DSR_ROBOT2 라이브러리 로드 실패")
        return

    _dsr_node.get_logger().info("단일 이동 명령 하달 (속도 제한: vel=5, acc=5)")
    _dsr_node.get_logger().info("이동이 시작되면 다른 터미널에서 즉시 MoveStop 서비스를 호출하십시오.")
    amovej(posj(J_HOME),vel=50,acc=50)
    # 단 한 번의 비동기 명령만 하달
    amovej(posj(J_WORK), vel=10, acc=10)

    try:
        # 노드 통신 상태를 유지하며 대기
        rclpy.spin(_dsr_node)
    except KeyboardInterrupt:
        _dsr_node.get_logger().info("사용자 종료 요청")
    finally:
        _dsr_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()