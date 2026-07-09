from rehab_assistant.exercises import ShoulderFlexionAnalyzer
from rehab_assistant.geometry import Point3D


def test_shoulder_flexion_scores_good_raised_arm():
    analyzer = ShoulderFlexionAnalyzer(side="left")
    landmarks = {
        "left_hip": Point3D(0, 1),
        "left_shoulder": Point3D(0, 0),
        "left_elbow": Point3D(1, 0),
        "left_wrist": Point3D(2, 0),
    }

    result = analyzer.analyze(landmarks)

    assert result.is_valid
    assert result.score >= 90
    assert result.phase == "up"


def test_shoulder_flexion_rejects_missing_visibility():
    analyzer = ShoulderFlexionAnalyzer(side="left")
    landmarks = {
        "left_hip": Point3D(0, 1),
        "left_shoulder": Point3D(0, 0),
        "left_elbow": Point3D(1, 0, visibility=0.1),
        "left_wrist": Point3D(2, 0),
    }

    result = analyzer.analyze(landmarks)

    assert not result.is_valid
    assert result.score == 0
