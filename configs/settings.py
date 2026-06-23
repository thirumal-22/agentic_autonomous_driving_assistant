import os

# YOLO Configuration
YOLO_MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.35

# Distance Calibration (Heuristic for demo purposes)
# Distance (m) = FOCAL_LENGTH_HEURISTIC / BBOX_HEIGHT_PIXELS
# Assuming average heights: Person = 1.7m, Car = 1.5m
DISTANCE_CALIBRATION = {
    "person": 150.0,    # factor to divide by normalized height or pixels
    "car": 250.0,
    "truck": 350.0,
    "motorcycle": 200.0,
    "stop sign": 120.0
}

# Distance Thresholds (Meters)
COLLISION_DISTANCE = 15.0  # Action: Stop
WARNING_DISTANCE = 35.0    # Action: Slow Down
SAFE_DISTANCE = 50.0       # Action: Maintain Speed

# Lane Detection Parameters
LANE_ROI_VERTICES = [
    (0.1, 0.95),   # Bottom-left (x, y) relative
    (0.45, 0.6),   # Top-left (x, y) relative
    (0.55, 0.6),   # Top-right (x, y) relative
    (0.9, 0.95)    # Bottom-right (x, y) relative
]
LANE_DEPARTURE_THRESHOLD = 0.15  # Distance deviation from center before warning

# Gemini Configuration
GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_SYSTEM_INSTRUCTION = """
You are the AI Reasoning Agent of an advanced autonomous driving assistant. 
Your job is to analyze the inputs from the Perception and Environment agents, 
and explain in natural language WHY a particular driving decision is recommended.
Provide a clear, human-readable, and concise explanation (1-2 sentences) detailing 
the detected obstacles, lane state, risks, and reasoning behind the decision.
Additionally, provide a 'safety_analysis' section and a 'final_recommendation'.
Output your reasoning in strict JSON format:
{
  "explanation": "Human-readable explanation.",
  "safety_analysis": "Key safety factors considered.",
  "final_recommendation": "Precise recommended throttle/steering action."
}
"""
