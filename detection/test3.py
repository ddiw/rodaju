import os
from ultralytics import YOLO

# --- [1. 학습 시 사용했던 설정값 그대로 적용] ---
MODEL_VARIANT = 'yolo11m-obb'
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
PROJECT_NAME = 'Recycle_Detection'
VERSION = 'v5'

# 실험 이름 및 저장 경로 구성
EXP_NAME = f"finetuned_{MODEL_VARIANT}_ep{EPOCHS}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{VERSION}"
SAVE_DIR = f"./runs/obb/{PROJECT_NAME}/{EXP_NAME}"

# 모델 경로 (학습 완료된 weight 위치)
MODEL_PATH = os.path.join(SAVE_DIR, 'weights', 'best.pt')

# 2. 모델 로드
if not os.path.exists(MODEL_PATH):
    print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
else:
    model = YOLO(MODEL_PATH)

    # 3. 검증 세트에 대해 추론 수행
    # project와 name을 변수로 넣어 폴더 구조를 학습 결과와 동기화합니다.
    results = model.predict(
        source=f'./integrated_dataset_{VERSION}/test/images',
        conf=0.25,
        save=True,
        project=f'{PROJECT_NAME}/{EXP_NAME}',      # 'Recycle_Detection' 폴더 내부에 생성
        name='inference', # 실험명_inference 폴더에 결과 저장
        exist_ok=True,
        imgsz=640 
    )

    # --- [4. 맨 아래에 FLOPs 및 모델 요약 출력] ---
    print("\n" + "="*60)
    print("📋 Final Model Complexity & Inference Summary")
    print("-" * 60)
    print(f"✅ 사용된 모델: {MODEL_VARIANT} ({EXP_NAME})")
    print(f"📂 결과 저장: {PROJECT_NAME}/{EXP_NAME}_inference")
    print(f"🖼️ 처리 이미지: {len(results)} 장")
    print("-" * 60)

    # 이 메서드가 터미널에 GFLOPs와 파라미터 수를 최종적으로 출력합니다.
    model.info(detailed=False, imgsz=640) 

    print("="*60 + "\n")
    print(f"🚀 모든 분석이 완료되었습니다!")

    # ... (기존 predict 코드 수행 완료 후) ...

# 4. Latency 및 FPS 계산 요약
speeds = [r.speed for r in results]
avg_preprocess = sum(s['preprocess'] for s in speeds) / len(speeds)
avg_inference = sum(s['inference'] for s in speeds) / len(speeds)
avg_postprocess = sum(s['postprocess'] for s in speeds) / len(speeds)
total_avg_ms = avg_preprocess + avg_inference + avg_postprocess

print("⏱️  Latency Analysis (Inference Speed)")
print("-" * 40)
print(f"1️⃣ 전처리(Pre-process):  {avg_preprocess:.2f} ms")
print(f"2️⃣ 추론(Inference):      {avg_inference:.2f} ms")
print(f"3️⃣ 후처리(Post-process): {avg_postprocess:.2f} ms")
print("-" * 40)
print(f"📊 총 지연시간(Total):   {total_avg_ms:.2f} ms / image")
print(f"🏃 초당 프레임(FPS):     {1000 / total_avg_ms:.2f} FPS")

# 마지막에 FLOPs 출력 (기존 코드)
model.info(imgsz=640)