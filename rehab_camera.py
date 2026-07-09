from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import cv2

from rehab_assistant.exercises import AnalysisResult, create_analyzer
from rehab_assistant.geometry import Point3D
from rehab_assistant.smoothing import AngleSmoother


LANDMARK_NAMES = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
}

POSE_CONNECTIONS = [
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe upper limb rehab prototype.")
    parser.add_argument(
        "--exercise",
        choices=["shoulder_flexion", "shoulder_abduction", "elbow_flexion"],
        default="shoulder_flexion",
        help="Exercise rule set to use.",
    )
    parser.add_argument("--side", choices=["left", "right"], default="left", help="Body side to analyze.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
    parser.add_argument(
        "--model",
        default="models/pose_landmarker_lite.task",
        help="PoseLandmarker .task model path, used when MediaPipe solutions API is unavailable.",
    )
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def mediapipe_landmarks_to_points(pose_landmarks: Any) -> dict[str, Point3D]:
    points: dict[str, Point3D] = {}
    for index, name in LANDMARK_NAMES.items():
        landmark = pose_landmarks.landmark[index]
        points[name] = Point3D(
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
            visibility=getattr(landmark, "visibility", 1.0),
        )
    return points


def tasks_landmarks_to_points(pose_landmarks: list[Any]) -> dict[str, Point3D]:
    points: dict[str, Point3D] = {}
    for index, name in LANDMARK_NAMES.items():
        landmark = pose_landmarks[index]
        points[name] = Point3D(
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
            visibility=getattr(landmark, "visibility", getattr(landmark, "presence", 1.0)),
        )
    return points


def draw_tasks_pose(frame, pose_landmarks: list[Any]) -> None:
    height, width = frame.shape[:2]
    for start, end in POSE_CONNECTIONS:
        a = pose_landmarks[start]
        b = pose_landmarks[end]
        ax, ay = int(a.x * width), int(a.y * height)
        bx, by = int(b.x * width), int(b.y * height)
        cv2.line(frame, (ax, ay), (bx, by), (80, 220, 120), 3, cv2.LINE_AA)

    for index in LANDMARK_NAMES:
        landmark = pose_landmarks[index]
        x, y = int(landmark.x * width), int(landmark.y * height)
        cv2.circle(frame, (x, y), 6, (40, 160, 255), -1, cv2.LINE_AA)


def smooth_result(result: AnalysisResult, smoother: AngleSmoother) -> AnalysisResult:
    smoothed = {
        name: smoother.update(f"{result.exercise}:{result.side}:{name}", value)
        for name, value in result.angles.items()
    }
    result.angles = smoothed
    return result


def draw_result(frame, result: AnalysisResult) -> None:
    color = (40, 180, 80) if result.score >= 80 else (40, 180, 240) if result.score >= 55 else (40, 80, 240)
    lines = [
        f"Exercise: {result.exercise} ({result.side})",
        f"Score: {result.score:.1f}   Reps: {result.rep_count}   Phase: {result.phase}",
    ]
    for name, value in result.angles.items():
        lines.append(f"{name.title()} angle: {value:.1f} deg")
    lines.extend(result.feedback[:2])

    x = 18
    y = 36
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        y += 32

    cv2.putText(frame, "Press Q or ESC to quit", (18, frame.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.putText(frame, "Press Q or ESC to quit", (18, frame.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)


def run() -> int:
    args = parse_args()

    try:
        import mediapipe as mp
    except ImportError:
        print("MediaPipe is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    analyzer = create_analyzer(args.exercise, args.side)
    smoother = AngleSmoother(window_size=5)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}.", file=sys.stderr)
        return 1

    try:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            return run_with_solutions(mp, args, analyzer, smoother, cap)
        return run_with_tasks(mp, args, analyzer, smoother, cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_with_solutions(mp: Any, args: argparse.Namespace, analyzer: Any, smoother: AngleSmoother, cap: cv2.VideoCapture) -> int:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    with mp_pose.Pose(
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the camera.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            detection = pose.process(rgb)
            rgb.flags.writeable = True

            if detection.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    detection.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )
                landmarks = mediapipe_landmarks_to_points(detection.pose_landmarks)
                result = analyzer.analyze(landmarks)
                draw_result(frame, smooth_result(result, smoother))
            else:
                cv2.putText(frame, "No body detected. Stand fully in frame.", (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 80, 240), 2, cv2.LINE_AA)

            cv2.imshow("Upper Limb Rehab - MediaPipe Prototype", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    return 0


def run_with_tasks(mp: Any, args: argparse.Namespace, analyzer: Any, smoother: AngleSmoother, cap: cv2.VideoCapture) -> int:
    model_path = Path(args.model)
    if not model_path.exists():
        print(
            "This MediaPipe install does not include mp.solutions.pose, so the newer Tasks API is required.\n"
            f"Missing model file: {model_path}\n\n"
            "Download pose_landmarker_lite.task and place it at models/pose_landmarker_lite.task:\n"
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            file=sys.stderr,
        )
        return 1

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    timestamp_ms = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the camera.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection = landmarker.detect_for_video(image, timestamp_ms)
            timestamp_ms += 33

            if detection.pose_landmarks:
                pose_landmarks = detection.pose_landmarks[0]
                draw_tasks_pose(frame, pose_landmarks)
                landmarks = tasks_landmarks_to_points(pose_landmarks)
                result = analyzer.analyze(landmarks)
                draw_result(frame, smooth_result(result, smoother))
            else:
                cv2.putText(frame, "No body detected. Stand fully in frame.", (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 80, 240), 2, cv2.LINE_AA)

            cv2.imshow("Upper Limb Rehab - MediaPipe Prototype", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
