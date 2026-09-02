import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.agent.pose_rules import check_pose
from app.cv.angle_utils import angle


def test_angle_straight_line_is_180():
    assert abs(angle((0, 0), (1, 0), (2, 0)) - 180) < 1e-3


def test_angle_right_angle_is_90():
    assert abs(angle((0, 1), (0, 0), (1, 0)) - 90) < 1e-3


def test_check_pose_unknown_asana():
    result = check_pose("not_a_real_pose", {})
    assert result["known"] is False


def test_check_pose_perfect_tree_pose():
    angles = {"left_knee": 170, "right_knee": 60, "left_hip": 175, "right_hip": 140}
    result = check_pose("tree_pose", angles)
    assert result["score"] == 1.0
    assert result["corrections"] == []
