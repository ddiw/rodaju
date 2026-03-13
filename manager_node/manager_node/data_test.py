import py_trees

# 1. 노드 A에서 값 쓰기
blackboard_writer = py_trees.blackboard.Client(name="Writer")
blackboard_writer.register_key(key="critical_data", access=py_trees.common.Access.WRITE)
blackboard_writer.critical_data = [1, 2, 3]

# 2. 노드 A를 강제로 삭제 (메모리 해제)
del blackboard_writer

# 3. 새로운 노드 B에서 값 읽기
blackboard_reader = py_trees.blackboard.Client(name="Reader")
blackboard_reader.register_key(key="critical_data", access=py_trees.common.Access.READ)

print(blackboard_reader.critical_data) 
# 결과는 [1, 2, 3]이 그대로 나옴. 이게 "Global Blackboard"의 힘이야.