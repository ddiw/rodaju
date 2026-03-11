import os
import shutil
from PIL import Image
import random # 데이터 섞기를 위해 추가

def integrate_yolo_datasets(src_root, dest_root, prefix, current_class_map, final_class_map, ref_size=None, max_count=None):
    """
    src_root: 각 데이터셋의 폴더 (images, labels 포함)
    dest_root: 저장될 통합 폴더
    prefix: 파일 중복 방지용 접두어 (예: 'ds1', 'ds2')
    current_class_map: 현재 이미지에서 확인한 클래스 순서 {번호: "이름"}
    final_class_map: 내가 통합할 클래스 순서 {"이름": 번호}
    """
    img_dest = os.path.join(dest_root, "valid/images")
    lbl_dest = os.path.join(dest_root, "valid/labels")
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(lbl_dest, exist_ok=True)

    # 소스 경로 설정
    src_img_dir = os.path.join(src_root, "valid/images")
    src_lbl_dir = os.path.join(src_root, "valid/labels")
    
    # --- 폴더 존재 유무 체크 (없으면 스킵) ---
    if not os.path.exists(src_img_dir) or not os.path.exists(src_lbl_dir):
        return ref_size, 0 
    
    files_processed = 0
    img_list = os.listdir(src_img_dir)
    # 랜덤하게 섞어서 앞부분만 가져오면 데이터가 골고루 뽑힙니다.
    random.shuffle(img_list) 
    
    for img_name in img_list:
        # max_count에 도달하면 중단
        if max_count is not None and files_processed >= max_count:
            break
            
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        
        name_wo_ext, ext = os.path.splitext(img_name)
        src_img_path = os.path.join(src_img_dir, img_name)
        src_lbl_path = os.path.join(src_lbl_dir, name_wo_ext + ".txt")

        if not os.path.exists(src_lbl_path): continue

        # 1. 이미지 크기 검사 (PIL 사용)
        with Image.open(src_img_path) as img:
            this_size = img.size
            if ref_size and this_size != ref_size:
                print(f"❌ [크기 불일치] {img_name} (기준:{ref_size}, 현재:{this_size})")
                continue
            if ref_size is None: ref_size = this_size

        # 2. 라벨 파일 읽기 및 ID 변환
        valid_boxes = []
        with open(src_lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.split()
                if not parts: continue
                
                try:
                    old_id = int(parts[0])
                    class_name = current_class_map.get(old_id)
                    
                    if class_name in final_class_map:
                        parts[0] = str(final_class_map[class_name])
                        valid_boxes.append(" ".join(parts))
                except ValueError:
                    continue

        # 해당 이미지에 내가 찾는 물체가 하나라도 있는 경우에만 저장
        if valid_boxes:
            new_name = f"{prefix}_{name_wo_ext}"
            # 이미지 복사
            shutil.copy2(src_img_path, os.path.join(img_dest, new_name + ext))
            # 수정된 라벨 저장
            with open(os.path.join(lbl_dest, new_name + ".txt"), 'w') as f:
                f.write("\n".join(valid_boxes))
            files_processed += 1
            
    return ref_size, files_processed

# --- 설정 구간 ---

# 1. 최종적으로 만들 데이터셋 구조
FINAL_TARGET = {"plastic": 0, "can": 1, "paper":2}

# 2. 첫 번째 데이터셋 맵 (bearing, bolt, flange, gear, nut, spring...)
DS1_MAP = {0: "can", 1:"paper", 2:"plastic"}

# # 3. 두 번째 데이터셋 맵 (bolt, nut, screw, washer)
DS2_MAP = {0: "plastic",}

# 3. 두 번째 데이터셋 맵 (bolt, nut, screw, washer)
DS3_MAP = {0: "can", 1: "can"}

DS4_MAP = {0: "paper"}

DEST_DIR = "./integrated_dataset_v5"

# 실행 (size가 계속 전달되므로 Unpacking 에러가 발생하지 않습니다)
size = None
size, count1 = integrate_yolo_datasets("./My-First-Project-3", DEST_DIR, "ds1", DS1_MAP, FINAL_TARGET, ref_size=size)
size, count2 = integrate_yolo_datasets("./dataset/bottle-2", DEST_DIR, "ds2", DS2_MAP, FINAL_TARGET, ref_size=size, max_count=800)
# ds3(paper)만 500장으로 제한합니다 (1장당 객체가 많으므로 500장만 해도 약 1,500~2,000개 인스턴스가 생성됨).
size, count3 = integrate_yolo_datasets("./dataset/can-2", DEST_DIR, "ds3", DS3_MAP, FINAL_TARGET, ref_size=size, max_count=700)
size, count4 = integrate_yolo_datasets("./dataset/paper-cup-3", DEST_DIR, "ds4", DS4_MAP, FINAL_TARGET, ref_size=size, max_count=650)


total_count = count1 + count2 + count3 + count4
print(f"\n✅ 통합 완료! 총 {total_count}개의 이미지-라벨 쌍이 생성되었습니다.")