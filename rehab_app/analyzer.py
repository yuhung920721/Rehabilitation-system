from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import calc_angle


VISIBILITY_THRESHOLD = 0.5

LM = {
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
}

ACTION_TEXT = {
    "S0": "起始位置",
    "S1": "手臂抬起中",
    "S2": "手臂已抬高",
    "NO_POSE": "未偵測到人體",
    "LOW_VISIBILITY": "關節辨識不清",
}


@dataclass
class UpdownStatus:
    action: str = "S0"
    action_text: str = "起始位置"
    angle: float | None = None
    score: float = 0.0
    rep_stage: str = "WAIT_S1"
    total: int = 0
    target_count: int = 5
    progress: float = 0.0
    s1_done: bool = False
    s2_done: bool = False
    feedback: str = "請站到鏡頭前，讓肩膀、手肘與髖部清楚入鏡。"
    is_finished: bool = False


@dataclass
class UpdownAnalyzer:
    angle_s0_max: float = 30.0
    angle_s1_max: float = 120.0
    ideal_s2_angle: float = 170.0
    target_count: int = 5
    rep_stage: str = "WAIT_S1"
    total: int = 0
    sum_score: float = 0.0
    rep_best_score: float = 0.0
    last_status: UpdownStatus = field(default_factory=UpdownStatus)

    def reset(self) -> None:
        self.rep_stage = "WAIT_S1"
        self.total = 0
        self.sum_score = 0.0
        self.rep_best_score = 0.0
        self.last_status = UpdownStatus(target_count=self.target_count)

    def set_target_count(self, target_count: int) -> None:
        self.target_count = max(1, min(50, target_count))
        self.reset()

    def analyze(self, landmarks: list[object] | None, image_width: int, image_height: int) -> UpdownStatus:
        if not landmarks:
            return self._status("NO_POSE", None, 0.0, "請站到鏡頭前，並讓上半身完整入鏡。")

        candidates: list[float] = []
        for side in ("RIGHT", "LEFT"):
            angle = self._get_arm_raise_angle(landmarks, side, image_width, image_height)
            if angle is not None:
                candidates.append(angle)

        if not candidates:
            return self._status("LOW_VISIBILITY", None, 0.0, "請露出肩膀、手肘與髖部，避免手臂離開畫面。")

        angle = max(candidates)
        action = self._classify(angle)
        current_score = max(0.0, 100.0 - abs(self.ideal_s2_angle - angle) / self.ideal_s2_angle * 100.0)
        self._update_rep_state(action, current_score)

        feedback = self._feedback(action, angle)
        status = self._status(action, angle, current_score, feedback)
        self.last_status = status
        return status

    def _get_arm_raise_angle(
        self,
        landmarks: list[object],
        side: str,
        image_width: int,
        image_height: int,
    ) -> float | None:
        hip, hip_visibility = self._get_landmark_xy(landmarks, LM[f"{side}_HIP"], image_width, image_height)
        shoulder, shoulder_visibility = self._get_landmark_xy(landmarks, LM[f"{side}_SHOULDER"], image_width, image_height)
        elbow, elbow_visibility = self._get_landmark_xy(landmarks, LM[f"{side}_ELBOW"], image_width, image_height)

        if min(hip_visibility, shoulder_visibility, elbow_visibility) < VISIBILITY_THRESHOLD:
            return None

        return calc_angle(hip, shoulder, elbow)

    @staticmethod
    def _get_landmark_xy(
        landmarks: list[object],
        index: int,
        image_width: int,
        image_height: int,
    ) -> tuple[tuple[float, float], float]:
        landmark = landmarks[index]
        visibility = getattr(landmark, "visibility", getattr(landmark, "presence", 1.0))
        return (landmark.x * image_width, landmark.y * image_height), float(visibility)

    def _classify(self, angle: float) -> str:
        if angle < self.angle_s0_max:
            return "S0"
        if angle < self.angle_s1_max:
            return "S1"
        return "S2"

    def _update_rep_state(self, action: str, current_score: float) -> None:
        if self.total >= self.target_count:
            return

        if self.rep_stage == "WAIT_S1":
            if action == "S1":
                self.rep_stage = "WAIT_S2"
                self.rep_best_score = current_score
        elif self.rep_stage == "WAIT_S2":
            self.rep_best_score = max(self.rep_best_score, current_score)
            if action == "S2":
                self.total += 1
                self.sum_score += self.rep_best_score
                self.rep_stage = "WAIT_RESET"
        elif self.rep_stage == "WAIT_RESET":
            if action == "S0":
                self.rep_stage = "WAIT_S1"
                self.rep_best_score = 0.0

    def _feedback(self, action: str, angle: float) -> str:
        if self.total >= self.target_count:
            return "訓練完成，可以按下重置再開始下一組。"
        if self.rep_stage == "WAIT_S1":
            return "從起始位置慢慢抬高手臂。"
        if self.rep_stage == "WAIT_S2":
            return "繼續抬高手臂，直到達到上方目標。"
        if self.rep_stage == "WAIT_RESET":
            return "很好，請慢慢放回起始位置。"
        return ACTION_TEXT.get(action, f"目前角度 {angle:.1f} 度")

    def _status(self, action: str, angle: float | None, score: float, feedback: str) -> UpdownStatus:
        progress = (self.total / self.target_count) * 100.0
        s1_done = self.rep_stage in ("WAIT_S2", "WAIT_RESET")
        s2_done = self.rep_stage == "WAIT_RESET"
        if self.total >= self.target_count:
            s1_done = True
            s2_done = True

        if self.total > 0:
            average_score = self.sum_score / self.total
        else:
            average_score = score

        return UpdownStatus(
            action=action,
            action_text=ACTION_TEXT.get(action, action),
            angle=angle,
            score=round(average_score, 1),
            rep_stage=self.rep_stage,
            total=self.total,
            target_count=self.target_count,
            progress=round(progress, 1),
            s1_done=s1_done,
            s2_done=s2_done,
            feedback=feedback,
            is_finished=self.total >= self.target_count,
        )

    def get_status(self) -> dict[str, object]:
        return self.last_status.__dict__.copy()
