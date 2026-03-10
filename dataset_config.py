import yaml
import os

# 통합 데이터셋 루트 경로 (본인의 경로에 맞게 수정)
dataset_root = os.path.abspath("./integrated_dataset_v5")

data_config = {
    "train": os.path.join(dataset_root, "train/images"),
    "val": os.path.join(dataset_root, "valid/images"),
    "test": os.path.join(dataset_root, "test/images"), # valid를 test로 합쳤으니 test 경로 지정
    "nc": 3,
    "names": ["plastic", "can", "paper"]
}

yaml_path = os.path.join(dataset_root, "data.yaml")
with open(yaml_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"✅ data.yaml 생성 완료: {yaml_path}")