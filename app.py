import os
import streamlit as st
import numpy as np
from PIL import Image
from utils.helpers import load_image

# Defer importing OpenCV to utility modules at runtime to avoid import-time
# native library failures (e.g., missing libGL) that can crash the app on
# some deployment images. Utilities in `utils/` handle the absence of cv2
# and provide fallbacks where possible.
from dotenv import load_dotenv

# Load local environment files if any
load_dotenv()

# Import cv2 at runtime with fallback; some deployments lack compatible wheels
try:
    import cv2
except Exception:
    cv2 = None

# Note: If cv2 is missing at runtime the UI will still work; features that
# rely on CV functions will either be no-ops or use PIL fallbacks from
# `utils/helpers.py` and `utils/cv_helpers.py`.

# Import graph orchestration
from graph import build_agent_graph

# Lazy import for cv_helpers to avoid triggering OpenCV import at module load time
draw_perception_overlay = None

# Set Premium Page Layout
st.set_page_config(
    page_title="Agentic Autonomous Driving Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling (Glassmorphism & Cyberpunk accents)
st.markdown("""
<style>
    .main {
        background-color: #0f111a;
        color: #e2e8f0;
    }
    .stApp {
        background-color: #0d0e15;
    }
    .metric-card {
        background: rgba(26, 29, 46, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .agent-header {
        font-weight: bold;
        color: #38bdf8;
        font-size: 1.1rem;
        margin-bottom: 5px;
        display: flex;
        align-items: center;
    }
    .agent-box {
        background: rgba(22, 28, 45, 0.7);
        border-left: 4px solid #38bdf8;
        padding: 15px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
    }
    .safety-alert {
        border-left: 4px solid #ef4444 !important;
        background: rgba(45, 20, 20, 0.6) !important;
    }
    .safety-ok {
        border-left: 4px solid #22c55e !important;
        background: rgba(20, 45, 20, 0.6) !important;
    }
    h1, h2, h3 {
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONFIG -----------------
st.sidebar.image("assets/logo.png", width=70)
st.sidebar.title("ADAS Control Panel")

# Gemini Key Input
gemini_key = st.sidebar.text_input(
    "Gemini API Key (Optional)",
    type="password",
    value=os.getenv("GEMINI_API_KEY", ""),
    help="Provide your Gemini API Key to enable real-time generative reasoning. Fallback template is used if left blank."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Perception Settings")
yolo_conf = st.sidebar.slider("YOLOv8 Conf Threshold", 0.1, 0.9, 0.35, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Driving Scenarios")
scenario_option = st.sidebar.selectbox(
    "Select Preloaded Sample Scenario",
    ["Select scenario...", "Clear Highway Traffic", "Pedestrian Crosswalk Warning", "Stop Sign Junction", "Lane Departure Warning"]
)

# Upload File Option
uploaded_file = st.sidebar.file_uploader("Or Upload Custom Road Image", type=["jpg", "jpeg", "png"])

# Define scenario paths
SCENARIO_MAP = {
    "Clear Highway Traffic": os.path.join(os.path.dirname(__file__), "data", "samples", "highway_traffic.png"),
    "Pedestrian Crosswalk Warning": os.path.join(os.path.dirname(__file__), "data", "samples", "pedestrian_crossing.png"),
    "Stop Sign Junction": os.path.join(os.path.dirname(__file__), "data", "samples", "stop_sign_junction.png"),
    "Lane Departure Warning": os.path.join(os.path.dirname(__file__), "data", "samples", "lane_drifting.png")
}

# ----------------- MAIN DASHBOARD -----------------
st.markdown("<h1 style='text-align: center; margin-bottom: 2px;'>Agentic Autonomous Driving Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem;'>Industry-Grade Multi-Agent Decision & Explainable AI (XAI) System</p>", unsafe_allow_html=True)
st.markdown("---")

# Determine Image Source
selected_img_path = None
uploaded_img = None

if uploaded_file is not None:
    # Read uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    if cv2 is not None:
        uploaded_img = cv2.imdecode(file_bytes, 1)
    else:
        # Fallback: use PIL to decode
        from io import BytesIO
        pil_img = Image.open(BytesIO(file_bytes)).convert('RGB')
        uploaded_img = np.array(pil_img)[:, :, ::-1]  # Convert RGB to BGR
elif scenario_option != "Select scenario...":
    selected_img_path = SCENARIO_MAP.get(scenario_option)
else:
    # Set default
    selected_img_path = SCENARIO_MAP["Clear Highway Traffic"]

# Run pipeline when image is ready
if selected_img_path or uploaded_img is not None:
    if uploaded_img is not None:
        raw_img = uploaded_img
        # Convert BGR to RGB
        if cv2 is not None:
            raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        else:
            raw_img_rgb = raw_img[:, :, ::-1] if len(raw_img.shape) == 3 else raw_img
    else:
        # Use helper with fallback support
        try:
            raw_img = load_image(selected_img_path)
        except Exception as e:
            st.error(f"Failed to load image: {e}")
            st.stop()
        if cv2 is not None:
            raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        else:
            raw_img_rgb = raw_img[:, :, ::-1] if len(raw_img.shape) == 3 else raw_img
        
    # 2. Compile and Run LangGraph
    graph = build_agent_graph()
    
    # Initialize State
    initial_state = {
        "frame_path": selected_img_path if uploaded_img is None else "",
        "image_data": raw_img,
        "gemini_api_key": gemini_key
    }
    
    with st.spinner("Executing Multi-Agent Coordination Graph..."):
        # Run graph
        final_state = graph.invoke(initial_state)
        
    # Extract results
    perception_results = final_state.get("perception_results", {})
    environment_context = final_state.get("environment_context", {})
    proposed_decision = final_state.get("proposed_decision", {})
    reasoning = final_state.get("reasoning", {})
    validated_decision = final_state.get("validated_decision", {})
    processed_img = final_state.get("processed_image")
    
    # Draw perception bounding boxes with updated risk levels
    # Import drawing helper lazily to prevent import-time OpenCV errors
    if draw_perception_overlay is None:
        try:
            from utils.cv_helpers import draw_perception_overlay as _dpo
            draw_perception_overlay = _dpo
        except Exception:
            draw_perception_overlay = None
            # Log lazy-import failure for debugging
            try:
                import traceback, datetime
                logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
                os.makedirs(logs_dir, exist_ok=True)
                log_path = os.path.join(logs_dir, f'cv_helpers_import_error_{datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}.log')
                with open(log_path, 'w') as f:
                    f.write('cv_helpers lazy import failed with the following traceback:\n')
                    traceback.print_exc(file=f)
            except Exception:
                pass

    if draw_perception_overlay is not None:
        annotated_img = draw_perception_overlay(processed_img.copy(), perception_results.get("detections", []))
    else:
        annotated_img = processed_img.copy()
    # Convert BGR to RGB for display
    if cv2 is not None:
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    else:
        annotated_img_rgb = annotated_img[:, :, ::-1] if len(annotated_img.shape) == 3 else annotated_img
    
    # ----------------- SECTION 1: SYSTEM STATE METRICS -----------------
    col1, col2, col3, col4 = st.columns(4)
    
    # Value styling
    final_act = validated_decision.get("final_action", "N/A")
    act_color = "#38bdf8"
    if final_act == "Stop Vehicle":
        act_color = "#ef4444"
    elif final_act == "Slow Down":
        act_color = "#f97316"
    elif "Change Lane" in final_act:
        act_color = "#eab308"
    else:
        act_color = "#22c55e"
        
    risk_score = environment_context.get("risk_score", 0.0)
    risk_color = "#22c55e"
    if risk_score > 7.0:
        risk_color = "#ef4444"
    elif risk_score > 4.0:
        risk_color = "#f97316"
        
    safety_status = validated_decision.get("safety_status", "N/A")
    safety_color = "#22c55e" if safety_status == "APPROVED" else "#ef4444"
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Autonomous Action</div>
            <div class="metric-value" style="color: {act_color};">{final_act}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Collision Risk Rating</div>
            <div class="metric-value" style="color: {risk_color};">{risk_score} <span style="font-size:1.2rem; color:#94a3b8;">/ 10</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Safety Enforcer Status</div>
            <div class="metric-value" style="color: {safety_color};">{safety_status}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Decision Conf. Score</div>
            <div class="metric-value" style="color: #a855f7;">{proposed_decision.get('confidence', 0.0) * 100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ----------------- SECTION 2: VISUAL CANVAS -----------------
    st.subheader("🤖 Computer Vision Canvas")
    view_col1, view_col2 = st.columns(2)
    
    with view_col1:
        st.image(raw_img_rgb, caption="Raw Road Camera Feed", use_container_width=True)
        
    with view_col2:
        st.image(annotated_img_rgb, caption="ADAS Perception Layer (YOLOv8 Objects + OpenCV Lane Overlay)", use_container_width=True)
        
    st.markdown("---")
    
    # ----------------- SECTION 3: AGENTIC WORKFLOW TIMELINE -----------------
    st.subheader("⛓️ Multi-Agent Orchestration & Reasoning Log (LangGraph Flow)")
    
    tab1, tab2 = st.columns([2, 1])
    
    with tab1:
        # Step 1: Perception Agent Log
        st.markdown(f"""
        <div class="agent-box">
            <div class="agent-header">🌐 Perception Agent (OpenCV & YOLOv8)</div>
            <strong>Lane Deviation:</strong> {perception_results.get('lane_metrics', {}).get('deviation', 0.0):.4f}<br>
            <strong>Lane Warning:</strong> {perception_results.get('lane_warning', 'Normal')}<br>
            <strong>Detections:</strong> {len(perception_results.get('detections', []))} objects recognized.
        </div>
        """, unsafe_allow_html=True)
        
        # Step 2: Environment Understanding Agent Log
        closest = environment_context.get("closest_obstacle")
        closest_str = f"{closest['class_name'].capitalize()} at {closest['distance']}m" if closest else "None"
        st.markdown(f"""
        <div class="agent-box">
            <div class="agent-header">🗺️ Environment Understanding Agent</div>
            <strong>Road Surface Hazard State:</strong> {environment_context.get('road_condition')}<br>
            <strong>Closest Immediate Threat:</strong> {closest_str}<br>
            <strong>Pedestrian Path Intrusion:</strong> {"YES (Collision Risk)" if environment_context.get('pedestrian_danger') else "No hazard detected"}
        </div>
        """, unsafe_allow_html=True)
        
        # Step 3: Decision-Making Agent Log
        st.markdown(f"""
        <div class="agent-box">
            <div class="agent-header">🧠 Decision-Making Agent (Safety Rules)</div>
            <strong>Proposed Maneuver:</strong> {proposed_decision.get('action')}<br>
            <strong>Initial Reasoning Proposal:</strong> {proposed_decision.get('rationale')}<br>
            <strong>Decision Stability Score:</strong> {proposed_decision.get('confidence')}
        </div>
        """, unsafe_allow_html=True)
        
        # Step 4: Safety Validation Agent Log
        safety_box_style = "agent-box safety-ok" if safety_status == "APPROVED" else "agent-box safety-alert"
        st.markdown(f"""
        <div class="{safety_box_style}">
            <div class="agent-header" style="color: {'#22c55e' if safety_status == 'APPROVED' else '#ef4444'};">🛡️ Safety Validation Agent (Failsafe Enforcer)</div>
            <strong>Validation Outcome:</strong> {safety_status}<br>
            <strong>Failsafe Reason/Logs:</strong> {validated_decision.get('safety_message')}
        </div>
        """, unsafe_allow_html=True)
        
    with tab2:
        # Generative AI Reasoning Panel
        st.markdown("<h4 style='margin-top:0;'>💡 Explainable AI (XAI) Panel</h4>", unsafe_allow_html=True)
        
        is_llm = reasoning.get("is_llm_generated", False)
        llm_badge = "🟢 Gemini-1.5-Flash Active" if is_llm else "⚪ Fallback Local Engine"
        st.info(f"Reasoning Core: {llm_badge}")
        
        st.markdown("##### 📝 Natural Language Explanation")
        st.write(reasoning.get("explanation"))
        
        st.markdown("##### 🔬 LLM Safety Analysis")
        st.write(reasoning.get("safety_analysis"))
        
        st.markdown("##### 🎯 Final Control Recommendations")
        st.success(reasoning.get("final_recommendation"))

# ----------------- PROJECT OVERVIEW FOOTER -----------------
st.markdown("---")
st.subheader("📖 Project Architecture & Technical Overview")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    ### System Orchestration
    This system utilizes a **directed acyclic graph (DAG)** orchestrated via **LangGraph** to coordinate multi-agent processes.
    By separating concerns, the architecture ensures that CV outputs are translated into hazard matrices, processed by safety rules, verified by an LLM reasoning engine, and validated against hard real-time safety constraints before steering commands are issued.
    
    ### Tech Stack Breakdown
    - **Computer Vision Pipeline**: OpenCV (Canny edge detection, Hough Line Transforms, ROI masking) combined with **YOLOv8** deep learning detector.
    - **State Orchestration**: **LangGraph** manages the state transitions and ensures clean separation of agents.
    - **Reasoning Engine**: **Gemini 1.5 Flash** for high-performance visual-context explanation.
    - **Safety Envelope**: Deterministic override system safeguarding against LLM hallucination or threshold corner cases.
    """)
with col_b:
    st.markdown("""
    ### Candidate Portfolio Description (ATS-Ready)
    *Developed an agentic autonomous driving assistant using Python, LangGraph, YOLOv8, OpenCV, and Gemini API. Designed a DAG-based state machine containing Perception, Environment, Decision, Reasoning, and Safety agents. Engineered an OpenCV lane tracking system generating departure warnings alongside YOLO-based pedestrian and sign distance models. Implemented an explainable AI pipeline utilizing Gemini 1.5 to output real-time driving justifications while safeguarding actions via a deterministic safety enforcer layer.*
    """)
