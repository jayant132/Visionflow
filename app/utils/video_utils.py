"""Frame sampling + skeleton drawing helpers (no heavy live-webrtc stack —
uploaded video is read frame-by-frame with OpenCV, sampled every Nth frame)."""
import cv2
import numpy as np

# minimal skeleton connections (subset of MediaPipe POSE_CONNECTIONS) for drawing
_CONNECTIONS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"), ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"), ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"), ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"), ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
]


def iter_sampled_frames(video_path: str, stride: int = 5):
    """Yield (frame_index, BGR frame) for every `stride`-th frame."""
    cap = cv2.VideoCapture(video_path)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def draw_landmarks(frame: np.ndarray, pts: dict) -> np.ndarray:
    """Draw skeleton overlay from a {name: (x, y, visibility)} landmark dict
    (normalized 0-1 coords), no mediapipe drawing-utils dependency needed."""
    annotated = frame.copy()
    if not pts:
        return annotated
    h, w = frame.shape[:2]

    def px(name):
        x, y, _ = pts[name]
        return int(x * w), int(y * h)

    for a, b in _CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(annotated, px(a), px(b), (255, 140, 0), 2)
    for name in pts:
        cv2.circle(annotated, px(name), 3, (0, 200, 0), -1)
    return annotated
