from rehab_assistant.geometry import Point3D, angle_between_points, score_range


def test_angle_between_points_right_angle():
    angle = angle_between_points(Point3D(1, 0), Point3D(0, 0), Point3D(0, 1))
    assert round(angle, 2) == 90.0


def test_angle_between_points_straight_line():
    angle = angle_between_points(Point3D(-1, 0), Point3D(0, 0), Point3D(1, 0))
    assert round(angle, 2) == 180.0


def test_score_range():
    assert score_range(90, 80, 100, 20) == 100.0
    assert score_range(70, 80, 100, 20) == 50.0
    assert score_range(40, 80, 100, 20) == 0.0
