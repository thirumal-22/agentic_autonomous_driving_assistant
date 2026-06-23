# Interview Talking Points & FAQs

Use these concise bullets during interviews to clearly explain your project, decisions, and results.

## Elevator Pitch (30s)
"I built an Agentic Autonomous Driving Assistant — a modular multi-agent system that combines YOLOv8-based perception, an environment risk scorer, a rule-based decision engine, and a deterministic safety enforcer. It also integrates a generative reasoning agent to provide human-readable explanations for decisions while ensuring safety-critical actions are validated locally before being issued."

## Key Contributions
- Designed a multi-agent DAG-based orchestration for clear separation of concerns.
- Implemented a hybrid perception stack (classical OpenCV lane detection + YOLOv8 object detection).
- Built a deterministic safety layer to validate and override LLM-generated suggestions.
- Integrated explainability using Gemini while ensuring operational fallbacks.

## Metrics to Mention
- Detection throughput (fps) on `yolov8n` model and average inference latency.
- False positive rates for critical classes (pedestrian, vehicle) on sample dataset.
- Safety override incidents per 1,000 frames (demonstrate conservative tuning).

## Common Interview Questions & Suggested Answers

Q: Why mix classic CV with deep learning?
- A: Classic CV (lane detection) is deterministic, low-latency, and explainable for structured features like lane markings. YOLOv8 provides robust detection for varied objects. Combining both leverages strengths and reduces dependency on only learned features.

Q: How do you ensure the LLM doesn’t produce unsafe instructions?
- A: The Safety Agent applies deterministic checks (distance thresholds, relative velocities, blind-spot verification) and can veto any action the LLM suggests. The LLM’s outputs are used primarily for explainability and context, not for direct control.

Q: How would you make this production-ready?
- A: Add hardware-in-the-loop simulation, stronger temporal fusion of detections (tracking), continuous integration with scenario-based testing, latency budgets, and formal verification for safety-critical primitives.

Q: What would you improve next?
- A: Add a perception confidence fusion module, better uncertainty estimation (e.g., MC-dropout), a richer benchmark dataset, and a replay tool for post-hoc safety analyses.

## Demo Script (quick run)
1. Start the app: `streamlit run app.py`
2. Open the local URL shown in the terminal.
3. Select a sample scenario and explain: perception outputs, risk score, decision candidate, and any safety overrides.


