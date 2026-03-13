import py_trees
import py_trees_ros
import rclpy

# 1. 커스텀 액션 노드 정의
class DoWork(py_trees.behaviour.Behaviour):
    def update(self):
        # 실제 로직 (예: 정렬, 이동 명령 등)
        print("작업 수행 중...")
        return py_trees.common.Status.SUCCESS

def main():
    rclpy.init()
    
    # 2. BT를 구동할 ROS 2 노드 생성
    node = rclpy.create_node("bt_manager")
    
    # 3. 트리 구성 (Root -> Sequence -> Children)
    root = py_trees.composites.Sequence(name="MainSequence", memory=True)
    
    # 예: 배터리 체크(내비게이션용 내장 노드 활용 가능) 및 작업 노드 추가
    work_node = DoWork(name="Work")
    root.add_child(work_node)
    
    # 4. ROS 2 전용 트리 매니저 설정
    # 이 매니저가 ROS 2의 Executor와 BT의 Tick을 연결합니다.
    tree = py_trees_ros.trees.BehaviourTree(root)
    tree.setup(node=node, timeout=15.0)
    
    # 5. 무한 루프 실행 (Tick-Tock)
    tree.tick_tock(period_ms=1000) # 1초마다 트리 전체를 한 번 훑음
    
    rclpy.spin(node)


if __name__ == "__main__":
    main()