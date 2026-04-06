import os
import base64
import tempfile

import cv2
import torch
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from torchvision import transforms

from models.cnn_lstm_model import CNN_LSTM


SEQ_LEN = 20

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

MIME_MAP = {
    "mp4":   "video/mp4",
    "avi":   "video/x-msvideo",
    "mov":   "video/quicktime",
    "mpeg4": "video/mp4",
}


st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #020811 !important;
    font-family: 'Rajdhani', sans-serif;
    color: #c8d8f0;
}

/* Blue radial mesh + grid */
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse 70% 50% at 15% 15%, rgba(30,90,255,0.12) 0%, transparent 55%),
        radial-gradient(ellipse 55% 70% at 85% 85%, rgba(0,180,255,0.10) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 60% 5%,  rgba(100,60,255,0.07) 0%, transparent 50%),
        linear-gradient(180deg, #020811 0%, #040d1a 50%, #020d14 100%);
    pointer-events: none; z-index: 0;
}
.stApp::after {
    content: '';
    position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(30,100,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(30,100,255,0.03) 1px, transparent 1px);
    background-size: 44px 44px;
    pointer-events: none; z-index: 0;
}

.block-container {
    padding: 2rem 3rem !important;
    max-width: 1280px !important;
    position: relative; z-index: 1;
}

#MainMenu, footer, header { visibility: hidden; }

/* Typography */
h1 {
    font-family: 'Orbitron', monospace !important;
    font-weight: 900 !important;
    font-size: clamp(1.6rem, 3.5vw, 2.6rem) !important;
    background: linear-gradient(135deg, #ffffff 0%, #60a5fa 45%, #3b82f6 70%, #1d4ed8 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: 0.04em !important;
    line-height: 1.2 !important;
}
h2 {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.2rem !important; font-weight: 600 !important;
    color: #93c5fd !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
}
h3 {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.8rem !important; font-weight: 700 !important;
    color: #3b82f6 !important;
    text-transform: uppercase !important; letter-spacing: 0.18em !important;
}
p, li, label, .stMarkdown p {
    color: #7a9cc0 !important;
    font-size: 0.9rem !important; line-height: 1.65 !important;
}
hr {
    border: none !important;
    border-top: 1px solid rgba(59,130,246,0.15) !important;
    margin: 1.75rem 0 !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(10,25,60,0.5) !important;
    border: 1.5px dashed rgba(59,130,246,0.4) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    background: rgba(30,58,138,0.2) !important;
    border-color: rgba(96,165,250,0.7) !important;
    box-shadow: 0 0 30px rgba(59,130,246,0.12) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: rgba(147,197,253,0.5) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(37,99,235,0.15) !important;
    border: 1px solid rgba(59,130,246,0.4) !important;
    border-radius: 8px !important;
    color: #93c5fd !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(37,99,235,0.35) !important;
    border-color: #60a5fa !important;
    box-shadow: 0 0 14px rgba(59,130,246,0.3) !important;
}

/* Analyze Button */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 50%, #1e3a8a 100%) !important;
    color: #e0f2fe !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    border: 1px solid rgba(96,165,250,0.4) !important;
    border-radius: 10px !important;
    padding: 0.85rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 0 24px rgba(29,78,216,0.35), inset 0 1px 0 rgba(255,255,255,0.08) !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 50%, #1e40af 100%) !important;
    border-color: rgba(147,197,253,0.7) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 0 45px rgba(59,130,246,0.45), inset 0 1px 0 rgba(255,255,255,0.12) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(10,20,50,0.7) !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.3rem !important;
    backdrop-filter: blur(10px) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(96,165,250,0.4) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.65rem !important;
    color: #60a5fa !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Rajdhani', sans-serif !important;
    color: #334d6e !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
}

/* Progress */
[data-testid="stProgressBar"] > div {
    background: rgba(30,58,138,0.3) !important;
    border-radius: 999px !important;
    height: 6px !important;
    overflow: hidden !important;
    border: 1px solid rgba(59,130,246,0.15) !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #1d4ed8, #60a5fa, #38bdf8) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 16px rgba(96,165,250,0.6) !important;
    transition: width 1.2s cubic-bezier(0.16,1,0.3,1) !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.9rem 1.1rem !important;
    letter-spacing: 0.02em !important;
    backdrop-filter: blur(8px) !important;
}

/* Spinner */
[data-testid="stSpinner"] > div {
    border-color: rgba(59,130,246,0.2) !important;
    border-top-color: #60a5fa !important;
}

/* Video */
video {
    border-radius: 12px !important;
    width: 100% !important;
    outline: none !important;
    display: block !important;
}

/* Columns */
[data-testid="column"] { padding: 0 0.6rem !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #020811; }
::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.3); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(96,165,250,0.5); }
</style>
""", unsafe_allow_html=True)


def render_video(video_path: str) -> None:
    if not os.path.exists(video_path):
        st.error(f"File not found: {video_path}")
        return

    ext  = video_path.rsplit(".", 1)[-1].lower()
    mime = MIME_MAP.get(ext, "video/mp4")

    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    st.markdown(f"""
    <div style="border-radius:14px; overflow:hidden;
                border:1px solid rgba(59,130,246,0.25);
                box-shadow:0 0 40px rgba(29,78,216,0.2), 0 8px 32px rgba(0,0,0,0.7);
                background:#000; position:relative;">
        <div style="position:absolute; top:0; left:0; right:0; height:2px;
                    background:linear-gradient(90deg,transparent,#3b82f6,#60a5fa,transparent);
                    z-index:2;"></div>
        <video controls preload="metadata"
               style="width:100%; max-height:440px; display:block; background:#000;">
            <source src="data:{mime};base64,{b64}" type="{mime}">
            Your browser does not support HTML5 video.
        </video>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model() -> CNN_LSTM:
    model = CNN_LSTM()
    model.load_state_dict(torch.load("deepfake_model.pth", map_location="cpu"))
    model.eval()
    return model



def extract_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Unable to open video. This can happen when OpenCV lacks the codec for the file format. Try converting the video to a standard H.264 mp4.")
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        st.error("Video file has no readable frames. Check that the video is not corrupted and is a supported format.")
        return None

    step = max(total // SEQ_LEN, 1)
    frames, idx = [], 0

    while cap.isOpened() and len(frames) < SEQ_LEN:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(TRANSFORM(Image.fromarray(frame)))
        idx += 1

    cap.release()
    if len(frames) < SEQ_LEN:
        st.warning(f"Video too short for reliable inference ({len(frames)} frames extracted, {SEQ_LEN} required).")
        return None
    return torch.stack(frames).unsqueeze(0)



def predict(video_path: str):
    frames = extract_frames(video_path)
    if frames is None:
        return "too_short", 0.0, 0.0, 0.0

    with torch.no_grad():
        prob = torch.sigmoid(load_model()(frames)).item()

    fake_prob  = prob
    real_prob  = 1.0 - prob
    label      = "FAKE" if prob > 0.5 else "REAL"
    confidence = fake_prob if label == "FAKE" else real_prob
    return label, confidence, real_prob, fake_prob



def make_chart(real_prob: float, fake_prob: float) -> go.Figure:
    fig = go.Figure(data=[
        go.Bar(
            x=["REAL", "FAKE"],
            y=[real_prob, fake_prob],
            marker=dict(
                color=["rgba(37,99,235,0.8)", "rgba(220,38,38,0.8)"],
                line=dict(color=["#60a5fa", "#f87171"], width=1.5),
            ),
            text=[f"{real_prob*100:.1f}%", f"{fake_prob*100:.1f}%"],
            textposition="outside",
            textfont=dict(family="Share Tech Mono", size=13, color="#93c5fd"),
            width=0.38,
        )
    ])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,20,50,0.4)",
        font=dict(color="#7a9cc0", family="Rajdhani"),
        yaxis=dict(
            title="Probability",
            range=[0, 1.25],
            tickformat=".0%",
            gridcolor="rgba(59,130,246,0.08)",
            zeroline=False,
            tickfont=dict(family="Share Tech Mono", size=11, color="#334d6e"),
        ),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(family="Orbitron", size=11, color="#60a5fa"),
        ),
        margin=dict(t=20, b=10, l=10, r=10),
        height=260,
        bargap=0.5,
        showlegend=False,
    )
    return fig



st.markdown("""
<div style="display:flex; align-items:center; gap:14px;
            margin-bottom:0.6rem; margin-top:0.5rem;">
    <div style="width:38px; height:38px; border-radius:10px;
                background:linear-gradient(135deg,#1d4ed8,#0ea5e9);
                display:flex; align-items:center; justify-content:center;
                box-shadow:0 0 20px rgba(29,78,216,0.5); font-size:1.1rem;">
        🧠
    </div>
    <span style="font-family:'Share Tech Mono',monospace; font-size:0.7rem;
                 color:#3b82f6; background:rgba(29,78,216,0.1);
                 border:1px solid rgba(59,130,246,0.25);
                 padding:3px 14px; border-radius:999px; letter-spacing:0.14em;">
        AI FORENSICS · CNN-LSTM · v1.0
    </span>
</div>
""", unsafe_allow_html=True)

st.title("DEEPFAKE DETECTION")

st.markdown("""
<p style="font-size:0.95rem; color:#334d6e; margin-top:0.2rem; margin-bottom:1.5rem;
          font-family:'Rajdhani',sans-serif; letter-spacing:0.03em;">
    Upload a video — CNN-LSTM analyses temporal frame patterns
    to detect AI-generated facial manipulation.
</p>
""", unsafe_allow_html=True)

st.markdown("---")


with st.spinner("Initialising model weights…"):
    model = load_model()


uploaded_video = st.file_uploader(
    label="drop video here",
    type=["mp4", "avi", "mov", "mpeg4"],
    label_visibility="collapsed",
    help="MP4 · AVI · MOV · MPEG4   ·   Max 200 MB",
)

st.markdown("""
<p style="text-align:center; font-size:0.72rem; color:#1e3a5f;
          font-family:'Share Tech Mono',monospace;
          margin-top:0.5rem; letter-spacing:0.1em;">
    DRAG & DROP  ·  MP4  AVI  MOV  MPEG4  ·  MAX 200 MB
</p>
""", unsafe_allow_html=True)



if uploaded_video is not None:

    # Save to disk (needed for OpenCV)
    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, uploaded_video.name)
    raw      = uploaded_video.getbuffer()

    with open(tmp_path, "wb") as fh:
        fh.write(raw)

    size_mb = len(raw) / (1024 * 1024)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # File info bar
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:14px;
                background:rgba(10,20,50,0.6);
                border:1px solid rgba(59,130,246,0.2);
                border-radius:10px; padding:10px 18px; margin-bottom:1.4rem;
                backdrop-filter:blur(10px);">
        <div style="width:32px; height:32px; border-radius:8px;
                    background:rgba(29,78,216,0.2);
                    border:1px solid rgba(59,130,246,0.3);
                    display:flex; align-items:center; justify-content:center;
                    font-size:1rem; flex-shrink:0;">🎬</div>
        <span style="font-family:'Share Tech Mono',monospace; font-size:0.82rem;
                     color:#60a5fa; flex:1;
                     overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
            {uploaded_video.name}
        </span>
        <span style="font-family:'Share Tech Mono',monospace; font-size:0.75rem;
                     color:#1e3a5f; white-space:nowrap; flex-shrink:0;">
            {size_mb:.1f} MB
        </span>
        <div style="width:8px; height:8px; border-radius:50%;
                    background:#22c55e; box-shadow:0 0 8px #22c55e; flex-shrink:0;">
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Video + controls layout
    vid_col, ctrl_col = st.columns([3, 1], gap="large")

    with vid_col:
        st.markdown("### 📽 Video Preview")
        render_video(tmp_path)

    with ctrl_col:
        st.markdown("### ⚙ Controls")

        # Info panel
        st.markdown(f"""
        <div style="background:rgba(10,20,50,0.6);
                    border:1px solid rgba(59,130,246,0.15);
                    border-radius:10px; padding:14px 16px;
                    margin-bottom:1rem; backdrop-filter:blur(10px);">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                        color:#1e3a5f; text-transform:uppercase; letter-spacing:0.14em;
                        margin-bottom:10px; border-bottom:1px solid rgba(59,130,246,0.1);
                        padding-bottom:8px;">
                File Info
            </div>
            <div style="display:flex; flex-direction:column; gap:9px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.78rem; color:#334d6e; font-family:'Rajdhani',sans-serif;">Format</span>
                    <span style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#60a5fa;">
                        {uploaded_video.name.rsplit(".",1)[-1].upper()}
                    </span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.78rem; color:#334d6e; font-family:'Rajdhani',sans-serif;">Size</span>
                    <span style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#60a5fa;">
                        {size_mb:.2f} MB
                    </span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.78rem; color:#334d6e; font-family:'Rajdhani',sans-serif;">Frames</span>
                    <span style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#60a5fa;">
                        {SEQ_LEN} sampled
                    </span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.78rem; color:#334d6e; font-family:'Rajdhani',sans-serif;">Model</span>
                    <span style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#60a5fa;">
                        CNN-LSTM
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        run_analysis = st.button("⚡  RUN ANALYSIS")

    st.markdown("---")

    # ── INFERENCE ──────────────────────────
    if run_analysis:
        with st.spinner("Extracting frames · Running CNN-LSTM inference…"):
            label, confidence, real_prob, fake_prob = predict(tmp_path)

        res_col, chart_col = st.columns([1, 1], gap="large")

        with res_col:
            st.markdown("### 🎯 Prediction Result")

            if label == "REAL":
                st.success("🟢  REAL VIDEO — No manipulation detected")
            elif label == "FAKE":
                st.error("🔴  FAKE VIDEO — Deepfake manipulation detected")
            else:
                st.warning("⚠️  Video too short — increase frame count")

            if label in ("REAL", "FAKE"):
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                st.metric("Confidence Score", f"{confidence * 100:.2f}%")
                st.progress(int(confidence * 100))
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Real", f"{real_prob * 100:.1f}%")
                with m2:
                    st.metric("Fake", f"{fake_prob * 100:.1f}%")

                # Verdict card
                is_real = label == "REAL"
                vc  = "#3b82f6"  if is_real else "#ef4444"
                vbg = "rgba(29,78,216,0.07)"  if is_real else "rgba(153,27,27,0.07)"
                vbd = "rgba(59,130,246,0.25)" if is_real else "rgba(239,68,68,0.25)"
                icon = "✔" if is_real else "✘"

                st.markdown(f"""
                <div style="margin-top:1.2rem; padding:18px 20px;
                            background:{vbg};
                            border:1px solid {vbd};
                            border-radius:12px; text-align:center;
                            position:relative; overflow:hidden;">
                    <div style="position:absolute; top:0; left:0; right:0; height:2px;
                                background:linear-gradient(90deg,transparent,{vc},transparent);">
                    </div>
                    <div style="font-family:'Orbitron',monospace; font-size:1.9rem;
                                font-weight:900; color:{vc}; letter-spacing:0.12em;
                                text-shadow:0 0 30px {vc}88;">
                        {icon}  {label}
                    </div>
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.62rem;
                                color:#1e3a5f; letter-spacing:0.18em;
                                text-transform:uppercase; margin-top:6px;">
                        FINAL VERDICT · {confidence*100:.1f}% CONFIDENCE
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with chart_col:
            st.markdown("### 📊 Distribution")
            st.plotly_chart(make_chart(real_prob, fake_prob), use_container_width=True)

            # Confidence ring
            pct  = int(confidence * 100)
            gc   = "#3b82f6" if label == "REAL" else "#ef4444"
            glow = "rgba(59,130,246,0.4)" if label == "REAL" else "rgba(239,68,68,0.4)"
            deg  = int(pct * 3.6)

            st.markdown(f"""
            <div style="display:flex; flex-direction:column;
                        align-items:center; margin-top:0.75rem; gap:10px;">
                <div style="width:120px; height:120px; border-radius:50%;
                            background:conic-gradient(
                                {gc} 0deg {deg}deg,
                                rgba(30,58,138,0.3) {deg}deg 360deg
                            );
                            display:flex; align-items:center; justify-content:center;
                            box-shadow:0 0 36px {glow}, 0 0 70px {glow}44;">
                    <div style="width:88px; height:88px; background:#020811;
                                border-radius:50%;
                                border:1px solid rgba(59,130,246,0.1);
                                display:flex; align-items:center; justify-content:center;
                                flex-direction:column; gap:2px;">
                        <span style="font-family:'Orbitron',monospace;
                                     font-size:1.3rem; font-weight:700; color:{gc};
                                     text-shadow:0 0 20px {gc};">
                            {pct}%
                        </span>
                        <span style="font-size:0.54rem; color:#1e3a5f;
                                     letter-spacing:0.14em; text-transform:uppercase;
                                     font-family:'Share Tech Mono',monospace;">
                            CONF.
                        </span>
                    </div>
                </div>
                <span style="font-family:'Share Tech Mono',monospace;
                             font-size:0.65rem; color:#1e3a5f;
                             letter-spacing:0.16em; text-transform:uppercase;">
                    CONFIDENCE RING
                </span>
            </div>
            """, unsafe_allow_html=True) 