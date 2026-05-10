import os
import cv2
import numpy as np
import config as cfg

# ================= 설정 섹션 =================
# 입력 경로 (이미지가 들어있는 폴더들)
IN_DIR_A = os.path.join(cfg.DATA_PATH, 'prediction')
IN_DIR_B = os.path.join(cfg.DATA_PATH, 'result', 'Figure', 'Figure_4', 'Figure_4_b')
IN_DIR_C = os.path.join(cfg.DATA_PATH, 'result', 'Figure', 'Figure_4', 'Figure_4_c_ext')

# 출력 경로
BASE_OUT = os.path.join(cfg.DATA_PATH, 'result', 'Figure', 'Figure_4')
OUT_A = os.path.join(BASE_OUT, 'Figure_4_a')
OUT_B = os.path.join(BASE_OUT, 'Figure_4_b_white')
OUT_C = os.path.join(BASE_OUT, 'Figure_4_c_ext_white')

# Center Line의 BGR 색상 (ID: 1)
CENTER_LINE_COLOR = (255, 77, 77)

for p in [OUT_A, OUT_B, OUT_C]:
    os.makedirs(p, exist_ok=True)


def process_a_filter_center(img_path, save_path):
    """center_line 색상만 남기고 나머지(배경 포함)는 전부 흰색으로 변경"""
    img = cv2.imread(img_path)
    if img is None: return

    # 1. center_line 색상과 일치하지 않는 모든 픽셀 찾기
    # 주의: 미세한 보간(interpolation)이 있을 수 있으므로 np.all 사용
    not_center = ~np.all(img == CENTER_LINE_COLOR, axis=-1)

    # 2. 해당 영역을 모두 흰색으로 변경
    img[not_center] = [255, 255, 255]
    cv2.imwrite(save_path, img)


def process_background_white(img_path, save_path):
    """검은색(0,0,0) 배경만 흰색(255,255,255)으로 변경"""
    img = cv2.imread(img_path)
    if img is None: return

    black_pixels = np.all(img == [0, 0, 0], axis=-1)
    img[black_pixels] = [255, 255, 255]
    cv2.imwrite(save_path, img)


# ---------------------------------------------------
# 실행 섹션
# ---------------------------------------------------

# 작업 (a): Center Line 필터링
print("Processing Figure 4 (a): Filtering only center_line...")
if os.path.exists(IN_DIR_A):
    for f_name in os.listdir(IN_DIR_A):
        if f_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_a_filter_center(os.path.join(IN_DIR_A, f_name), os.path.join(OUT_A, f_name))

# 작업 (b) & (c): 배경색 반전
print("Processing Figure 4 (b) & (c): Inverting black background to white...")
for folder_in, folder_out, label in [(IN_DIR_B, OUT_B, "b"), (IN_DIR_C, OUT_C, "c")]:
    if not os.path.exists(folder_in):
        print(f"Skipping {label}: Folder not found.")
        continue
    for f_name in os.listdir(folder_in):
        if f_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_background_white(os.path.join(folder_in, f_name), os.path.join(folder_out, f_name))

print(f"Done! Figure 4 images are saved in: {BASE_OUT}")