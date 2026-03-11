import os
import shutil

def merge_validation_and_test(base_path):
    """
    base_path: 'integrated_dataset_v2' 폴더 경로
    valid 폴더의 데이터를 test 폴더로 통합합니다.
    """
    valid_img_dir = os.path.join(base_path, "valid/images")
    valid_lbl_dir = os.path.join(base_path, "valid/labels")
    test_img_dir = os.path.join(base_path, "test/images")
    test_lbl_dir = os.path.join(base_path, "test/labels")

    # 1. 목적지(test) 폴더가 없으면 생성
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_lbl_dir, exist_ok=True)

    # 2. 원본(valid) 폴더 존재 확인
    if not os.path.exists(valid_img_dir):
        print(f"ℹ️  옮길 원본 valid 폴더가 없습니다: {valid_img_dir}")
        return

    print(f"🚀 통합 시작: {valid_img_dir} -> {test_img_dir}")

    files_moved = 0 # 변수 이름 통일
    # valid 폴더에서 이미지 목록을 가져옴
    valid_files = [f for f in os.listdir(valid_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in valid_files:
        name_wo_ext, ext = os.path.splitext(img_name)
        lbl_name = name_wo_ext + ".txt"

        src_img = os.path.join(valid_img_dir, img_name)
        src_lbl = os.path.join(valid_lbl_dir, lbl_name)
        
        dest_img = os.path.join(test_img_dir, img_name)
        dest_lbl = os.path.join(test_lbl_dir, lbl_name)

        # 라벨 파일이 존재할 때만 이동
        if os.path.exists(src_lbl):
            # 목적지(test)에 이미 같은 이름의 파일이 있으면 이름 변경
            if os.path.exists(dest_img):
                new_name = f"from_valid_{img_name}"
                new_lbl_name = f"from_valid_{lbl_name}"
                dest_img = os.path.join(test_img_dir, new_name)
                dest_lbl = os.path.join(test_lbl_dir, new_lbl_name)

            # 파일 이동
            shutil.move(src_img, dest_img)
            shutil.move(src_lbl, dest_lbl)
            files_moved += 1
        else:
            print(f"⚠️  라벨이 없는 이미지 발견(제외됨): {img_name}")

    print(f"\n✅ 통합 완료! 총 {files_moved}세트의 데이터가 test 폴더로 합쳐졌습니다.")

# --- 실행 ---
path_v1 = "./integrated_dataset_v2"
merge_validation_and_test(path_v1)