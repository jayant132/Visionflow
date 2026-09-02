"""Rule-based correctness checker. Free, deterministic, low-latency —
this is the primary feedback path; an LLM is optional and only used
for natural-language phrasing (see adk_agent.py)."""

# ideal joint angle (deg) with tolerance, per asana
ASANA_RULES = {
    "tree_pose": {
        "left_knee": (170, 15), "right_knee": (60, 25),
        "left_hip": (175, 15), "right_hip": (140, 25),
    },
    "warrior2": {
        "left_knee": (100, 15), "right_knee": (170, 15),
        "left_hip": (150, 20), "right_hip": (170, 15),
    },
    "downward_dog": {
        "left_elbow": (170, 15), "right_elbow": (170, 15),
        "left_knee": (170, 15), "right_knee": (170, 15),
        "left_hip": (80, 20), "right_hip": (80, 20),
    },
}

def check_pose(asana: str, angles: dict) -> dict:
    rules = ASANA_RULES.get(asana)
    if not rules:
        return {"asana": asana, "known": False, "score": 0.0, "corrections": ["Unknown asana"]}

    corrections, hits = [], 0
    for joint, (ideal, tol) in rules.items():
        actual = angles.get(joint)
        if actual is None:
            continue
        if abs(actual - ideal) <= tol:
            hits += 1
        else:
            direction = "straighten" if actual < ideal else "bend"
            corrections.append(f"{direction} your {joint.replace('_', ' ')}")

    score = round(hits / len(rules), 2)
    return {"asana": asana, "known": True, "score": score, "corrections": corrections[:3]}
