from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .geometry import Point3D, angle_between_points, score_range


Landmarks = dict[str, Point3D]


@dataclass
class AnalysisResult:
    exercise: str
    side: str
    score: float
    rep_count: int
    phase: str
    angles: dict[str, float]
    feedback: list[str] = field(default_factory=list)
    is_valid: bool = True


class ExerciseAnalyzer(Protocol):
    name: str
    side: str
    rep_count: int
    phase: str

    def analyze(self, landmarks: Landmarks) -> AnalysisResult:
        ...


@dataclass
class BaseAnalyzer:
    side: str = "left"
    rep_count: int = 0
    phase: str = "start"
    min_visibility: float = 0.45

    @property
    def shoulder(self) -> str:
        return f"{self.side}_shoulder"

    @property
    def elbow(self) -> str:
        return f"{self.side}_elbow"

    @property
    def wrist(self) -> str:
        return f"{self.side}_wrist"

    @property
    def hip(self) -> str:
        return f"{self.side}_hip"

    def _has_required(self, landmarks: Landmarks, names: list[str]) -> bool:
        return all(name in landmarks and landmarks[name].visibility >= self.min_visibility for name in names)

    def _invalid(self, exercise: str, names: list[str]) -> AnalysisResult:
        missing = ", ".join(names)
        return AnalysisResult(
            exercise=exercise,
            side=self.side,
            score=0.0,
            rep_count=self.rep_count,
            phase=self.phase,
            angles={},
            feedback=[f"Please keep these joints visible: {missing}."],
            is_valid=False,
        )


@dataclass
class ShoulderFlexionAnalyzer(BaseAnalyzer):
    name: str = "shoulder_flexion"

    def analyze(self, landmarks: Landmarks) -> AnalysisResult:
        required = [self.hip, self.shoulder, self.elbow, self.wrist]
        if not self._has_required(landmarks, required):
            return self._invalid(self.name, required)

        shoulder_angle = angle_between_points(landmarks[self.hip], landmarks[self.shoulder], landmarks[self.elbow])
        elbow_angle = angle_between_points(landmarks[self.shoulder], landmarks[self.elbow], landmarks[self.wrist])

        if self.phase in {"start", "down"} and shoulder_angle >= 80:
            self.phase = "up"
        elif self.phase == "up" and shoulder_angle <= 35:
            self.phase = "down"
            self.rep_count += 1

        shoulder_score = score_range(shoulder_angle, 80, 130, 55)
        elbow_score = score_range(elbow_angle, 150, 180, 45)
        score = round(shoulder_score * 0.7 + elbow_score * 0.3, 1)

        feedback = []
        if shoulder_angle < 80:
            feedback.append("Raise the arm higher.")
        elif shoulder_angle > 130:
            feedback.append("Do not over-raise the arm.")
        if elbow_angle < 150:
            feedback.append("Keep the elbow straighter.")
        if not feedback:
            feedback.append("Good posture. Hold briefly, then return slowly.")

        return AnalysisResult(
            exercise=self.name,
            side=self.side,
            score=score,
            rep_count=self.rep_count,
            phase=self.phase,
            angles={"shoulder": shoulder_angle, "elbow": elbow_angle},
            feedback=feedback,
        )


@dataclass
class ShoulderAbductionAnalyzer(BaseAnalyzer):
    name: str = "shoulder_abduction"

    def analyze(self, landmarks: Landmarks) -> AnalysisResult:
        required = [self.hip, self.shoulder, self.elbow, self.wrist]
        if not self._has_required(landmarks, required):
            return self._invalid(self.name, required)

        shoulder_angle = angle_between_points(landmarks[self.hip], landmarks[self.shoulder], landmarks[self.elbow])
        elbow_angle = angle_between_points(landmarks[self.shoulder], landmarks[self.elbow], landmarks[self.wrist])

        if self.phase in {"start", "down"} and shoulder_angle >= 70:
            self.phase = "up"
        elif self.phase == "up" and shoulder_angle <= 30:
            self.phase = "down"
            self.rep_count += 1

        shoulder_score = score_range(shoulder_angle, 70, 110, 55)
        elbow_score = score_range(elbow_angle, 150, 180, 45)
        score = round(shoulder_score * 0.7 + elbow_score * 0.3, 1)

        feedback = []
        if shoulder_angle < 70:
            feedback.append("Move the arm outward and upward.")
        elif shoulder_angle > 110:
            feedback.append("Lower slightly; target is around shoulder height.")
        if elbow_angle < 150:
            feedback.append("Keep the elbow straighter.")
        if not feedback:
            feedback.append("Good range. Keep the trunk steady.")

        return AnalysisResult(
            exercise=self.name,
            side=self.side,
            score=score,
            rep_count=self.rep_count,
            phase=self.phase,
            angles={"shoulder": shoulder_angle, "elbow": elbow_angle},
            feedback=feedback,
        )


@dataclass
class ElbowFlexionAnalyzer(BaseAnalyzer):
    name: str = "elbow_flexion"

    def analyze(self, landmarks: Landmarks) -> AnalysisResult:
        required = [self.shoulder, self.elbow, self.wrist]
        if not self._has_required(landmarks, required):
            return self._invalid(self.name, required)

        elbow_angle = angle_between_points(landmarks[self.shoulder], landmarks[self.elbow], landmarks[self.wrist])

        if self.phase in {"start", "extended"} and elbow_angle <= 95:
            self.phase = "bent"
        elif self.phase == "bent" and elbow_angle >= 150:
            self.phase = "extended"
            self.rep_count += 1

        score = round(score_range(elbow_angle, 45, 95, 70), 1)

        feedback = []
        if elbow_angle > 95:
            feedback.append("Bend the elbow more.")
        elif elbow_angle < 45:
            feedback.append("Relax slightly; do not bend too far.")
        else:
            feedback.append("Good elbow bend. Return slowly.")

        return AnalysisResult(
            exercise=self.name,
            side=self.side,
            score=score,
            rep_count=self.rep_count,
            phase=self.phase,
            angles={"elbow": elbow_angle},
            feedback=feedback,
        )


def create_analyzer(exercise: str, side: str) -> ExerciseAnalyzer:
    analyzers = {
        "shoulder_flexion": ShoulderFlexionAnalyzer,
        "shoulder_abduction": ShoulderAbductionAnalyzer,
        "elbow_flexion": ElbowFlexionAnalyzer,
    }
    try:
        return analyzers[exercise](side=side)
    except KeyError as exc:
        choices = ", ".join(sorted(analyzers))
        raise ValueError(f"Unknown exercise '{exercise}'. Choose one of: {choices}") from exc
