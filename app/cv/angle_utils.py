import numpy as np

def angle(a, b, c) -> float:
    """Angle at point b (in degrees) formed by points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
