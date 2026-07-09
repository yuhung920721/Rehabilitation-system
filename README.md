# MediaPipe 上肢復健動作分析原型

這是一個先不架網站的 Python 原型，使用 MediaPipe 偵測人體節點，再計算肩、肘、腕等關節角度，用規則判斷上肢復健動作是否正確。

## 目前功能

- 即時攝影機偵測人體骨架
- 計算關節角度
- 顯示動作分數、次數、階段與修正提示
- 支援三個基礎動作：
  - `shoulder_flexion`：肩屈曲，手臂向前上舉
  - `shoulder_abduction`：肩外展，手臂向側邊抬起
  - `elbow_flexion`：肘屈曲，彎曲手肘

## 安裝

建議使用 Python 3.10 到 3.12。若使用 Python 3.13 或 3.14，MediaPipe 的舊版 `solutions` 介面可能不可用，程式會改用新版 Tasks API。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果你的 Windows 找不到 `python` 指令，可以改用 `py`：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

如果執行時看到 `mediapipe has no attribute solutions` 或提示缺少 `pose_landmarker_lite.task`，請下載新版 MediaPipe Tasks 模型：

```powershell
python scripts/download_pose_model.py
```

下載後會產生：

```text
models/pose_landmarker_lite.task
```

## 執行

左手肩屈曲：

```powershell
python rehab_camera.py --exercise shoulder_flexion --side left
```

右手肘屈曲：

```powershell
python rehab_camera.py --exercise elbow_flexion --side right
```

可選參數：

```powershell
python rehab_camera.py --exercise shoulder_abduction --side left --camera 0 --width 1280 --height 720
```

按 `Q` 或 `ESC` 結束程式。

## 測試

```powershell
pip install -r requirements-dev.txt
pytest
```

## 建議拍攝方式

- 使用者站在鏡頭前 1.5 到 2 公尺。
- 上半身、肩膀、手肘、手腕、髖部都要入鏡。
- 光線充足，避免衣服與背景顏色太接近。
- 肩屈曲建議側面或斜側面測試；肩外展建議正面測試。
- 這個系統適合作為復健輔助與專題展示，不可取代醫師或復健師診斷。

## 專案結構

```text
rehab_camera.py              # 攝影機即時原型入口
rehab_assistant/
  geometry.py                # 角度、分數等數學工具
  exercises.py               # 復健動作規則
  smoothing.py               # 角度平滑
tests/                       # 基本單元測試
```

## 後續可擴充方向

- 儲存每次復健紀錄為 CSV。
- 加入更多復健動作。
- 讓每個使用者設定個人化目標角度。
- 將角度序列拿來訓練分類模型，例如 SVM、Random Forest 或 LSTM。
- 未來再把核心模組接到 Flask 或前端網頁。
