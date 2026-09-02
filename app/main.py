"""Streamlit UI: upload a yoga video, get frame-sampled pose analysis +
AI coaching feedback, with live metrics/latency/logs visible in-app."""
import os
import tempfile
import time

import streamlit as st

from app.config import settings
from app.telemetry import (
    log, start_metrics_server, new_trace_id,
    FRAME_COUNTER, POSE_SCORE_HIST, CV_LATENCY, REQUESTS, timed,
)
from app.guardrails import validate_upload, rate_limiter, GuardrailError
from app.cv.pose_estimator import PoseEstimator
from app.cv.pose_classifier import classify, ASANAS
from app.utils.video_utils import iter_sampled_frames, draw_landmarks
from app.agent.adk_agent import get_feedback

st.set_page_config(page_title="AI Yoga Pose Coach", layout="wide")
start_metrics_server()

if "estimator" not in st.session_state:
    st.session_state.estimator = PoseEstimator(min_conf=settings.MIN_LANDMARK_CONFIDENCE)

st.title("🧘 AI Yoga Pose Coach")
st.caption("MediaPipe pose estimation + Keras classifier + Google ADK agent (local Ollama LLM) — fully local, fully free.")

with st.sidebar:
    st.header("Settings")
    asana_choice = st.selectbox("Target asana", ["auto-detect"] + ASANAS)
    stride = st.slider("Frame sample stride", 1, 15, settings.FRAME_SAMPLE_STRIDE)
    st.divider()
    st.caption(f"Metrics: http://localhost:{settings.METRICS_PORT}/metrics")
    llm_status = f"ON (Ollama: {settings.OLLAMA_MODEL})" if settings.USE_LLM_FEEDBACK else "OFF (rules only)"
    st.caption(f"LLM feedback: {llm_status}")

uploaded = st.file_uploader("Upload a short yoga clip (mp4/mov, ≤ %dMB)" % settings.MAX_UPLOAD_MB,
                             type=["mp4", "mov", "avi"])

if uploaded is not None:
    trace_id = new_trace_id()
    log.info("upload_received", trace_id=trace_id, filename=uploaded.name, size_kb=len(uploaded.getvalue()) // 1024)

    try:
        if not rate_limiter.allow():
            raise GuardrailError("rate_limit", "Too many requests — please wait a minute.")
        validate_upload(uploaded.getvalue(), uploaded.name)
    except GuardrailError as e:
        REQUESTS.labels(status="blocked").inc()
        st.error(f"⛔ {e.message}")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.getvalue())
        video_path = tmp.name

    frame_slot = st.empty()
    result_slot = st.empty()
    feedback_slot = st.empty()
    metrics_cols = st.columns(4)

    scores, latencies = [], []
    hint = None if asana_choice == "auto-detect" else asana_choice

    try:
        for idx, frame in iter_sampled_frames(video_path, stride=stride):
            with timed(CV_LATENCY, trace_id=trace_id, frame=idx):
                pts = st.session_state.estimator.process(frame)
                if pts is None:
                    continue
                angles = st.session_state.estimator.joint_angles(pts)
                analysis = classify(angles, asana_hint=hint)

            FRAME_COUNTER.inc()
            POSE_SCORE_HIST.observe(analysis["score"])
            scores.append(analysis["score"])

            annotated = draw_landmarks(frame, pts)
            frame_slot.image(annotated, channels="BGR", caption=f"Frame {idx}")

            with result_slot.container():
                c1, c2 = st.columns(2)
                c1.metric("Detected asana", analysis["asana"].replace("_", " ").title())
                c2.metric("Correctness score", f"{analysis['score']*100:.0f}%")
                if analysis.get("corrections"):
                    st.warning("Corrections: " + "; ".join(analysis["corrections"]))
                else:
                    st.success("Alignment looks good on this frame.")

            time.sleep(0.02)  # keep UI responsive

        if scores:
            feedback = get_feedback({"asana": hint or "detected_pose", "score": sum(scores) / len(scores),
                                      "corrections": analysis.get("corrections", [])})
            latencies.append(feedback["latency_ms"])
            with feedback_slot.container():
                st.subheader("🧠 AI Coach Feedback")
                st.info(feedback["text"])
                st.caption(f"source={feedback['source']} · latency={feedback['latency_ms']}ms")

            avg_score = sum(scores) / len(scores)
            metrics_cols[0].metric("Avg score", f"{avg_score*100:.0f}%")
            metrics_cols[1].metric("Frames analyzed", len(scores))
            metrics_cols[2].metric("Agent latency", f"{feedback['latency_ms']:.0f} ms")
            metrics_cols[3].metric("Trace ID", trace_id)

            REQUESTS.labels(status="success").inc()
            log.info("analysis_complete", trace_id=trace_id, avg_score=avg_score, frames=len(scores))
        else:
            st.warning("No pose detected in the uploaded video. Try a clearer, well-lit clip.")
            REQUESTS.labels(status="no_pose").inc()

    except Exception as e:
        REQUESTS.labels(status="error").inc()
        log.error("processing_failed", trace_id=trace_id, error=str(e))
        st.error(f"Something went wrong while processing the video: {e}")
    finally:
        os.unlink(video_path)
else:
    st.info("👆 Upload a yoga video to get started. Supported: Tree Pose, Warrior II, Downward Dog.")
