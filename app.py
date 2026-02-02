import streamlit as st
import google.generativeai as genai
import tempfile
import os
import time
from dotenv import load_dotenv

# =========================================================
# ENV + GEMINI CONFIG
# =========================================================
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found in .env file")
    st.stop()

genai.configure(api_key=API_KEY)

# Use Flash for low-latency demos (still Gemini 3 family)
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Explain the World",
    page_icon="👁️",
    layout="wide"
)

st.markdown("""
<style>
.stButton>button {
    width: 100%;
    background-color: #ff4b4b;
    color: white;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("👁️ Explain the World")
    st.markdown("---")
    st.markdown("**Status:** 🟢 Online")
    st.markdown(f"**Model:** {MODEL_NAME}")
    st.markdown("Powered by **Gemini 3 Multimodal Reasoning**")

    reasoning_mode = st.selectbox(
        "🎯 Analysis Mode",
        [
            "General Causal Analysis",
            "Industrial Safety",
            "Security Surveillance",
            "Sports Analytics"
        ]
    )

# =========================================================
# PROMPT ENGINE
# =========================================================
def get_system_prompt(mode: str) -> str:
    base_prompt = """
You are a Multimodal Video Reasoning Engine.

You are analyzing a video that represents a sequence of events unfolding over time.
Earlier events may explain later outcomes.

CRITICAL RULES:
- Base your reasoning ONLY on visible evidence.
- Do NOT assume unseen causes.
- If something is unclear, explicitly state uncertainty.

Your objective is to explain WHY events occur, not just WHAT is visible.

Respond in MARKDOWN using this format:

## 1. 🔍 Situation Summary
(Describe what is happening visually over time)

## 2. 🔗 Causal Chain
(Explain cause → effect using arrows)

## 3. ⚠️ Risk Assessment
(Identify anomalies, dangers, or likely failures)

## 4. 🛡️ Recommended Action
(Suggest immediate next steps)

## 5. 📊 Confidence Level
(High / Medium / Low + short justification)
"""

    domain_focus = {
        "Industrial Safety":
            "Focus on machinery behavior, conveyor jams, mechanical failures, and worker safety hazards.",
        "Security Surveillance":
            "Focus on suspicious motion, unauthorized access, abnormal behavior, or security threats.",
        "Sports Analytics":
            "Focus on player positioning, tactical breakdowns, momentum shifts, and likely next plays."
    }

    return base_prompt + "\n\n" + domain_focus.get(mode, "")

# =========================================================
# VIDEO ANALYSIS PIPELINE
# =========================================================
def analyze_video(uploaded_video, user_query):
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_video.read())
    temp_file.close()

    # Upload to Gemini
    with st.spinner("📤 Uploading video to Gemini..."):
        video_file = genai.upload_file(path=temp_file.name)

    # Wait for processing
    with st.spinner("⏳ Building temporal understanding..."):
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            st.error("❌ Video processing failed")
            return None

    # Gemini call
    model = genai.GenerativeModel(MODEL_NAME)

    prompt = (
        get_system_prompt(reasoning_mode)
        + "\n\nUser Question: "
        + user_query
    )

    response_stream = model.generate_content(
        [video_file, prompt],
        stream=True
    )

    return response_stream

# =========================================================
# UI
# =========================================================
st.title("🔥 Real-Time Multimodal ‘Explain the World’ Assistant")
st.markdown("**Point. Ask. Understand *why*.**")

col1, col2 = st.columns([1, 1])

with col1:
    st.info("📥 Input Video")
    video = st.file_uploader("Upload a short video (MP4 / MOV)", type=["mp4", "mov"])
    if video:
        st.video(video)

with col2:
    st.success("🧠 Reasoning Output")

    if video:
        if st.button("Analyze Causal Logic"):
            output_box = st.empty()
            full_output = ""

            try:
                stream = analyze_video(video, "Explain what happened and why.")

                if stream:
                    for chunk in stream:
                        if chunk.text:
                            full_output += chunk.text
                            output_box.markdown(full_output)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.write("Waiting for visual input…")

st.markdown("---")
st.caption("⚡ Built with Gemini 3 Multimodal Reasoning | Hackathon Prototype")
