"""Thin wrapper around MediaPipe Pose (free, on-device, no API calls)."""
import cv2
import numpy as np
import mediapipe as mp
from app.cv.angle_utils import angle

_mp_pose = mp.solutions.pose
LM = _mp_pose.PoseLandmark

class PoseEstimator:
    def __init__(self, min_conf: float = 0.5):
        self._pose = _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,          # 0/1/2: balance latency vs accuracy
            min_detection_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )

    def process(self, bgr_frame: np.ndarray):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None
        pts = {lm.name: (p.x, p.y, p.visibility)
               for lm, p in zip(LM, result.pose_landmarks.landmark)}
        return pts

    @staticmethod
    def joint_angles(pts: dict) -> dict:
        def xy(name): return pts[name][:2]
        return {
            "left_elbow": angle(xy("LEFT_SHOULDER"), xy("LEFT_ELBOW"), xy("LEFT_WRIST")),
            "right_elbow": angle(xy("RIGHT_SHOULDER"), xy("RIGHT_ELBOW"), xy("RIGHT_WRIST")),
            "left_knee": angle(xy("LEFT_HIP"), xy("LEFT_KNEE"), xy("LEFT_ANKLE")),
            "right_knee": angle(xy("RIGHT_HIP"), xy("RIGHT_KNEE"), xy("RIGHT_ANKLE")),
            "left_hip": angle(xy("LEFT_SHOULDER"), xy("LEFT_HIP"), xy("LEFT_KNEE")),
            "right_hip": angle(xy("RIGHT_SHOULDER"), xy("RIGHT_HIP"), xy("RIGHT_KNEE")),
        }

    def close(self):
        self._pose.close()
