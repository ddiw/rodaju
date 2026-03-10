import os
import shutil
from ultralytics import YOLO
from ultralytics import settings 

os.environ["WANDB_MODE"] = "disabled" # WandB 끄기
settings.update({"tensorboard": True})

# --- 학습 설정값 ---
MODEL_VARIANT = 'yolov8s-obb'
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001  # 92% 이상에서는 더 세밀한 튜닝을 위해 0.001에서 낮춤
PROJECT_NAME = 'Recycle_Detection'
VERSION = 'v5'

# tensorboard logging name
name_tag = "tensorboard_logging" # << tensorborad logging (jh added)

# 실험 이름 생성 (기존과 구분되도록 v3 또는 high_precision 추가 권장)
EXP_NAME = f"finetuned_{MODEL_VARIANT}_ep{EPOCHS}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{VERSION}"
SAVE_DIR = f"./{PROJECT_NAME}/{EXP_NAME}"

# 1. 모델 로드
model = YOLO(f'{MODEL_VARIANT}.pt')

# 2. 학습 실행 (정밀 튜닝 및 증강 강화 적용)
model.train(
    data=f'./integrated_dataset_{VERSION}/data.yaml',
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    seed=7541,
    lr0=LEARNING_RATE,
    lrf=0.01,             
    cos_lr=True,          
    imgsz=640,
    device=0,             
    optimizer='AdamW',    
    weight_decay=0.001,  # 규제를 조금 완화하여 데이터를 더 깊게 학습
    warmup_epochs=3.0,    
    
    # # --- 클래스 불균형 및 정밀도 향상 전략 ---
    # cls=1.0,              # 기본값(0.5)보다 높여 볼트/너트 오분류 방지 강화
    # box=7.5,              # 박스 위치 정확도 유지
    
    # # --- 증강(Augmentation) 대폭 강화 ---
    # mosaic=1.0,           
    # mixup=0.2,            # 0.15에서 0.2로 상향하여 복잡한 상황 학습 강화
    # copy_paste=0.1,       # [추가] 개별 물체를 복사/붙여넣기 하여 데이터 밀도 향상
    # degrees=15.0,         
    # scale=0.6,            # 다양한 거리에서의 물체 크기 학습
    
    project=PROJECT_NAME,
    name=EXP_NAME,
    exist_ok=True,
    workers=4,            
    cache=False,

    # save_period=1,  # 매 에폭마다 저장 강제
    # plots=True      # 학습 결과 그래프 생성 강제
)

# 3. 가중치 파일 이동
base_weight = f'{MODEL_VARIANT}.pt'
if os.path.exists(base_weight):
    os.makedirs(SAVE_DIR, exist_ok=True)
    shutil.move(base_weight, os.path.join(SAVE_DIR, base_weight))

# 4. 검증 실행 (TTA 제외, 기본 검증으로 확인)
metrics = model.val()
print(f"✅ 최종 mAP50: {metrics.box.map50}")
print(f"🚀 학습 완료! 결과 확인: {SAVE_DIR}")
