from __future__ import annotations

from pathlib import Path
from threading import Lock
from time import sleep, time
from typing import Iterator

import cv2
import mediapipe as mp

from .analyzer import LM, VISIBILITY_THRESHOLD, UpdownAnalyzer


POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
]


class CameraService:
    def __init__(self, camera_index: int = 0, model_path: str = "models/pose_landmarker_lite.task") -> None:
        self.camera_index = camera_index
        self.model_path = Path(model_path)
        self.analyzer = UpdownAnalyzer()
        self.lock = Lock()
        self._landmarker = None
        self._start_time = time()

    def get_status(self) -> dict[str, object]:
        with self.lock:
            status = self.analyzer.get_status()
            status["model_ready"] = self.model_path.exists()
            status["model_path"] = str(self.model_path)
            return status

    def reset(self) -> None:
        with self.lock:
            self.analyzer.reset()

    def set_target_count(self, target_count: int) -> None:
        with self.lock:
            self.analyzer.set_target_count(target_count)

    def frame_stream(self) -> Iterator[bytes]:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            while True:
                yield self._error_frame("Camera is unavailable")
                sleep(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        if not self.model_path.exists():
            while True:
                ok, frame = cap.read()
                if not ok:
                    frame = self._blank_frame("Camera frame unavailable")
                else:
                    frame = cv2.flip(frame, 1)
                    self._draw_banner(frame, "Missing MediaPipe model. Run: python scripts/download_pose_model.py")
                yield self._encode_frame(frame)
                sleep(0.05)

        landmarker = self._get_landmarker()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    yield self._error_frame("Camera frame unavailable")
                    sleep(0.2)
                    continue

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time() - self._start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    self._draw_pose_skeleton(frame, landmarks, width, height)
                    with self.lock:
                        self.analyzer.analyze(landmarks, width, height)
                else:
                    with self.lock:
                        self.analyzer.analyze(None, width, height)
                    self._draw_banner(frame, "No body detected")

                yield self._encode_frame(frame)
        finally:
            cap.release()

    def _get_landmarker(self):
        if self._landmarker is not None:
            return self._landmarker

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        return self._landmarker

    def _draw_pose_skeleton(self, frame, landmarks, image_width: int, image_height: int) -> None:
        for start, end in POSE_CONNECTIONS:
            point_a = landmarks[start]
            point_b = landmarks[end]
            if self._visibility(point_a) < VISIBILITY_THRESHOLD or self._visibility(point_b) < VISIBILITY_THRESHOLD:
                continue

            a = (int(point_a.x * image_width), int(point_a.y * image_height))
            b = (int(point_b.x * image_width), int(point_b.y * image_height))
            cv2.line(frame, a, b, (82, 211, 143), 3, cv2.LINE_AA)

        for index in LM.values():
            point = landmarks[index]
            if self._visibility(point) < VISIBILITY_THRESHOLD:
                continue
            center = (int(point.x * image_width), int(point.y * image_height))
            cv2.circle(frame, center, 6, (44, 123, 229), -1, cv2.LINE_AA)

    @staticmethod
    def _visibility(point) -> float:
        return float(getattr(point, "visibility", getattr(point, "presence", 1.0)))

    @staticmethod
    def _draw_banner(frame, text: str) -> None:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (20, 24, 31), -1)
        cv2.putText(frame, text, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 247, 250), 2, cv2.LINE_AA)

    @staticmethod
    def _blank_frame(text: str):
        frame = cv2.UMat(540, 960, cv2.CV_8UC3).get()
        frame[:] = (20, 24, 31)
        CameraService._draw_banner(frame, text)
        return frame

    def _error_frame(self, text: str) -> bytes:
        return self._encode_frame(self._blank_frame(text))

    @staticmethod
    def _encode_frame(frame) -> bytes:
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return b""
        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
