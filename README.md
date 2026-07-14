# 上肢復健動作輔助系統 Flask 版

這個版本使用 `Flask + HTML/CSS/JavaScript + OpenCV + MediaPipe Tasks API`。

畫面分成兩區：

- 左側：即時攝影機畫面，只顯示人物與 MediaPipe 骨架。
- 右側：APP 式操作面板，顯示目前動作、完成次數、兩個動作指示燈、角度、分數與總進度條。

## 目前動作

目前以你提供的 `Updown_mediapipe.py` 方法為主，做「手臂上下抬起」辨識。

系統會計算左右手的：

```text
髖部 - 肩膀 - 手肘
```

也就是以肩膀為中心的手臂抬起角度。

狀態分成：

```text
S0：起始位置，角度 < 30 度
S1：手臂抬起中，30 度 <= 角度 < 120 度
S2：手臂已抬高，角度 >= 120 度
```

完成一次的流程：

```text
S0 -> S1 -> S2 -> 回到 S0
```

其中右側兩個指示燈分別代表：

- 中段動作：是否達到 S1。
- 上舉動作：是否達到 S2。

## 安裝

建議使用 Python 3.10 到 3.12。

```powershell
cd C:\Users\USER\Project\Rehab
py -3.12 -m venv rehab
.\rehab\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果你的電腦沒有 Python 3.12，也可以先試：

```powershell
py -m venv rehab
.\rehab\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 下載 MediaPipe 模型

新版 MediaPipe Tasks API 需要 `pose_landmarker_lite.task`。

```powershell
python scripts/download_pose_model.py
```

下載完成後應該會有：

```text
models/pose_landmarker_lite.task
```

## 執行

```powershell
python app.py
```

接著打開瀏覽器：

```text
http://127.0.0.1:5000
```

## 專案結構

```text
app.py                       # Flask 入口
rehab_app/
  analyzer.py                # Updown 動作角度與次數判斷
  camera.py                  # 攝影機、MediaPipe、影像串流
  geometry.py                # 角度計算
templates/
  index.html                 # APP 主畫面
static/
  css/app.css                # 介面樣式
  js/app.js                  # 即時狀態更新
scripts/
  download_pose_model.py     # 下載 MediaPipe 模型
```

## 拍攝建議

- 半身即可，但要露出肩膀、手肘與髖部。
- 手臂不要超出畫面。
- 光線要充足。
- 完成上舉後要慢慢回到起始位置，系統才會準備計算下一次。

## API

狀態資料：

```text
GET /api/status
```

重置次數：

```text
POST /api/reset
```

設定目標次數：

```text
POST /api/target
```

範例資料：

```json
{
  "action": "S1",
  "action_text": "手臂抬起中",
  "angle": 86.5,
  "score": 72.1,
  "total": 2,
  "target_count": 5,
  "progress": 40.0,
  "s1_done": true,
  "s2_done": false
}
```
