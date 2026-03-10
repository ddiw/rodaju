from roboflow import Roboflow
rf = Roboflow(api_key="--")
project = rf.workspace("chaeyoung-lrujj").project("my-first-project-b62a7")
version = project.version(3)
dataset = version.download("yolov8-obb")
                