# Updown_mediapipe.py
# ------------------------------------------------------------
# 肩部上下舉（Updown）動作辨識 — MediaPipe Pose 版本
# 使用新版 MediaPipe Tasks API (PoseLandmarker)
# 取代原本 Teachable Machine (.h5) 整圖分類方式
#
# 感知（Perception） : MediaPipe PoseLandmarker 擷取 33 個關節座標
# 決策（Decision）    : 計算「髖—肩—肘」夾角，判斷 S0 / S1 / S2
# 行動（Action）      : 疊加示範圖、進度條、次數、得分即時回饋
# ------------------------------------------------------------

import os
import sys
import time
import traceback
import numpy as np
import cv2
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

# ============================================================
# 1. 基本設定：改用相對路徑，不管資料夾放在哪台電腦、叫什麼名字都能跑
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
fontpath = os.path.join(BASE_DIR, 'NotoSansTC-Regular.ttf')
image_path = os.path.join(BASE_DIR, 'Updown.png')
model_path = os.path.join(BASE_DIR, 'pose_landmarker_lite.task')

# ---- 防呆檢查 1：字型檔存不存在 ----
if not os.path.exists(fontpath):
    print(f"[錯誤] 找不到字型檔：{fontpath}")
    print("請確認 NotoSansTC-Regular.ttf 有放在跟這支 .py 檔同一層資料夾")
    sys.exit(1)
font = ImageFont.truetype(fontpath, 22)

# ---- 防呆檢查 2：PoseLandmarker 模型檔存不存在 ----
if not os.path.exists(model_path):
    print(f"[錯誤] 找不到 MediaPipe 模型檔：{model_path}")
    print("請先下載模型，執行以下指令：")
    print('Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile "pose_landmarker_lite.task"')
    sys.exit(1)

# ---- 示範動作圖片：選用，找不到就跳過疊圖，不中斷程式 ----
overlay_enabled = False
move_image = None

if os.path.exists(image_path):
    move_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if move_image is None:
        print(f"[警告] 找到檔案但讀取失敗：{image_path}（可能檔案損毀），將跳過疊圖")
    elif move_image.shape[2] != 4:
        print(f"[警告] {image_path} 沒有透明通道，將跳過疊圖")
        move_image = None
    else:
        new_width = 240
        aspect_ratio = move_image.shape[1] / move_image.shape[0]
        new_height = int(new_width / aspect_ratio)
        move_image = cv2.resize(move_image, (new_width, new_height))
        overlay_enabled = True
        print(f"[提示] 已載入示範圖片：{image_path}")
else:
    print(f"[提示] 找不到示範圖片 {image_path}，將跳過疊圖繼續執行核心辨識功能")

# ---- 防呆檢查 3：攝影機開不開得起來 ----
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[錯誤] 無法開啟攝影機 (index 0, CAP_DSHOW)")
    print("嘗試改用預設 backend...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[錯誤] 兩種 backend 皆無法開啟攝影機")
        print("請檢查：1) 是否有其他程式正在占用攝影機(如 Teams/瀏覽器分頁)  2) Windows 設定 > 隱私權 > 相機 是否允許桌面應用程式存取")
        sys.exit(1)

cap.set(3, 480)
cap.set(4, 320)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', 800, 600)

# ============================================================
# 2. MediaPipe PoseLandmarker 初始化 (Tasks API)
# ============================================================
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = PoseLandmarker.create_from_options(options)
start_time = time.time()

VISIBILITY_THRESHOLD = 0.5  # 關節點可信度低於此值視為「看不到」，避免誤判

# ---- BlazePose 33 個關節點的固定編號（Tasks API 不再提供具名列舉，改用數字索引）----
LM = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
}

# 拿來畫骨架示意用的簡化連線（原本 mp_drawing.draw_landmarks 也是壞掉的 solutions 模組功能，自己畫）
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]

# ============================================================
# 3. 角度計算核心 —— 這是「決策層」的靈魂
# ============================================================
def calc_angle(a, b, c):
    """
    計算 b 點（頂點）在 a-b-c 三點所形成的夾角，單位：度。
    a = 髖部 HIP、b = 肩膀 SHOULDER（頂點）、c = 手肘 ELBOW
    手臂垂放時角度接近 0°，側平舉時接近 90°，舉過頭時接近 170°~180°
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def get_landmark_xy(landmarks, idx, image_width, image_height):
    lm = landmarks[idx]
    return (lm.x * image_width, lm.y * image_height), lm.visibility


def get_arm_raise_angle(landmarks, side, image_width, image_height):
    """side: 'LEFT' 或 'RIGHT'"""
    hip, v1 = get_landmark_xy(landmarks, LM[f'{side}_HIP'], image_width, image_height)
    shoulder, v2 = get_landmark_xy(landmarks, LM[f'{side}_SHOULDER'], image_width, image_height)
    elbow, v3 = get_landmark_xy(landmarks, LM[f'{side}_ELBOW'], image_width, image_height)

    if min(v1, v2, v3) < VISIBILITY_THRESHOLD:
        return None, False

    angle = calc_angle(hip, shoulder, elbow)
    return angle, True


def draw_pose_skeleton(image, landmarks, image_width, image_height):
    """簡易骨架繪製，取代原本壞掉的 mp_drawing.draw_landmarks"""
    for idx_a, idx_b in POSE_CONNECTIONS:
        pa = landmarks[idx_a]
        pb = landmarks[idx_b]
        if pa.visibility < VISIBILITY_THRESHOLD or pb.visibility < VISIBILITY_THRESHOLD:
            continue
        pt_a = (int(pa.x * image_width), int(pa.y * image_height))
        pt_b = (int(pb.x * image_width), int(pb.y * image_height))
        cv2.line(image, pt_a, pt_b, (0, 255, 0), 2)
        cv2.circle(image, pt_a, 4, (0, 0, 255), -1)
        cv2.circle(image, pt_b, 4, (0, 0, 255), -1)


# ============================================================
# 4. 動作判斷門檻（需要你們自己錄影測試校準，這只是起始值）
# ============================================================
ANGLE_S0_MAX = 30
ANGLE_S1_MAX = 120
IDEAL_S2_ANGLE = 170

action_mapping = {"S0": "無動作", "S1": "肩部側舉", "S2": "肩部上舉"}

# ============================================================
# 5. 計數與狀態初始化
#
# 用「狀態機」管理一次完整動作的流程，強制規定必須依序經過：
#   WAIT_S1（等待側舉） → WAIT_S2（已側舉，等待上舉） → WAIT_RESET（已上舉，等待放下回到S0）
# 只有照這個順序走完一輪，才會 total += 1。
# 這樣可以避免角度在 S1/S2 門檻附近(120°上下)輕微晃動、雜訊跳動時被誤判成「又完成一次」。
# ============================================================
rep_stage = "WAIT_S1"
total = 0
target_percentage = 0
sum_score = 0
rep_best_score = 0  # 追蹤「這一次重複動作」過程中出現過的最佳分數
target_count = 5
predicted_label = "S0"
s1_done = False  # 僅用於畫面顯示(側舉完成的綠燈)，由 rep_stage 推算，不再自己單獨判斷
s2_done = False  # 僅用於畫面顯示(上舉完成的綠燈)，由 rep_stage 推算，不再自己單獨判斷

# ============================================================
# 6. 主迴圈
# ============================================================
try:
    while True:
        success, image = cap.read()
        if not success:
            print("[警告] 攝影機讀取失敗 (cap.read() 回傳 False)，程式即將結束")
            break

        if overlay_enabled:
            y_offset = image.shape[0] - move_image.shape[0] - 125
            x_offset = image.shape[1] - move_image.shape[1] - 400
            alpha_s = move_image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                image[y_offset:y_offset + move_image.shape[0], x_offset:x_offset + move_image.shape[1], c] = (
                    alpha_s * move_image[:, :, c] +
                    alpha_l * image[y_offset:y_offset + move_image.shape[0],
                                     x_offset:x_offset + move_image.shape[1], c]
                )

        image = cv2.flip(image, 1)
        image_height, image_width = image.shape[:2]

        # ---- MediaPipe Tasks API 需要 mp.Image 物件 ----
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        frame_timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        predicted_label_chinese = action_mapping.get(predicted_label, predicted_label)
        current_score = 0

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]  # 只取第一個偵測到的人

            angle_right, vis_right = get_arm_raise_angle(landmarks, 'RIGHT', image_width, image_height)
            angle_left, vis_left = get_arm_raise_angle(landmarks, 'LEFT', image_width, image_height)

            candidates = [a for a, v in [(angle_right, vis_right), (angle_left, vis_left)] if v]

            if candidates:
                angle = max(candidates)

                if angle < ANGLE_S0_MAX:
                    predicted_label = "S0"
                elif angle < ANGLE_S1_MAX:
                    predicted_label = "S1"
                else:
                    predicted_label = "S2"

                current_score = max(0, 100 - abs(IDEAL_S2_ANGLE - angle) / IDEAL_S2_ANGLE * 100)
                predicted_label_chinese = action_mapping.get(predicted_label, predicted_label)
                print(predicted_label, f"{angle:.1f}°", f"score={current_score:.1f}", f"stage={rep_stage}")

                draw_pose_skeleton(image, landmarks, image_width, image_height)
            else:
                predicted_label_chinese = "偵測不到手臂"
        else:
            predicted_label_chinese = "偵測不到人"

        # ---- 狀態機：強制規定「S0 → S1 → S2 → 回到S0」依序完成才算一次 ----
        if total < target_count:
            if rep_stage == "WAIT_S1":
                # 還沒側舉，持續等待，角度分數不列入計算(這階段的抖動不該影響分數)
                if predicted_label == "S1":
                    rep_stage = "WAIT_S2"
                    rep_best_score = current_score

            elif rep_stage == "WAIT_S2":
                # 已經側舉過，持續追蹤這次循環出現過的最佳分數，等待上舉
                rep_best_score = max(rep_best_score, current_score)
                if predicted_label == "S2":
                    total += 1
                    sum_score += rep_best_score
                    target_percentage = (total / target_count) * 100
                    rep_stage = "WAIT_RESET"
                    # 注意：這裡故意不重置 rep_best_score，讓畫面上的「上舉完成」燈維持到真正放下手臂為止

            elif rep_stage == "WAIT_RESET":
                # 已完成這一次(側舉+上舉都做到)，必須先放下手臂回到 S0，才能開始算下一次
                if predicted_label == "S0":
                    rep_stage = "WAIT_S1"
                    rep_best_score = 0

        # 畫面顯示用的兩個燈號，直接由目前狀態推算，不再獨立判斷
        s1_done = rep_stage in ("WAIT_S2", "WAIT_RESET")
        s2_done = rep_stage == "WAIT_RESET"

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        # ---- 文字版面：全部由上往下垂直堆疊，避免因畫面寬度不足造成左右文字重疊 ----
        draw.text((10, 10), '目前動作：{}'.format(predicted_label_chinese), fill=(0, 0, 0), font=font)
        draw.text((10, 36), '次數: {}'.format(total), fill=(0, 0, 0), font=font)

        row3_y = 68
        row4_y = 96
        circle_x = 165  # 圓點畫在文字後面，往右留一點空間

        draw.text((10, row3_y), '側舉完成：', fill=(0, 0, 0), font=font)
        draw.ellipse([(circle_x, row3_y + 2), (circle_x + 18, row3_y + 20)],
                     fill=(0, 255, 0) if s1_done else (0, 0, 255))

        draw.text((10, row4_y), '上舉完成：', fill=(0, 0, 0), font=font)
        draw.ellipse([(circle_x, row4_y + 2), (circle_x + 18, row4_y + 20)],
                     fill=(0, 255, 0) if s2_done else (0, 0, 255))

        if total >= target_count:
            score = sum_score / target_count
            # ---- 中文字一律用 PIL 畫，cv2.putText 不支援中文，會變成問號亂碼 ----
            draw.text((10, 130), '目標已達成！總得分：{:.1f}分'.format(score), fill=(0, 0, 255), font=font)

        progress_bar_width = int((target_percentage / 100) * pil_image.width)
        draw.rectangle([(0, pil_image.height - 26), (progress_bar_width, pil_image.height)], fill=(138, 217, 255))
        draw.text((10, pil_image.height - 50), '完成進度：{:.2f}%'.format(target_percentage), fill=(0, 0, 0), font=font)

        image = np.array(pil_image)
        cv2.imshow("Frame", image)

        if cv2.waitKey(1) != -1:
            break

except Exception as e:
    print("\n[未預期的錯誤] 程式發生例外，完整錯誤如下：")
    traceback.print_exc()
    print(f"\n錯誤摘要：{e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("[提示] 程式已結束，攝影機資源已釋放")