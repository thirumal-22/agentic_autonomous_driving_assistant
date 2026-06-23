from typing import TypedDict, Dict, Any, List
import numpy as np
from langgraph.graph import StateGraph, END

# Import agent classes
from agents.perception_agent import PerceptionAgent
from agents.environment_agent import EnvironmentAgent
from agents.decision_agent import DecisionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.safety_agent import SafetyAgent

# Define the State representing the autonomous driving system data stream
class DrivingState(TypedDict):
    frame_path: str
    image_data: np.ndarray          # Image frame currently being evaluated
    processed_image: np.ndarray      # Image with drawn lane lines/bounding boxes
    perception_results: Dict[str, Any]
    environment_context: Dict[str, Any]
    proposed_decision: Dict[str, Any]
    reasoning: Dict[str, Any]
    validated_decision: Dict[str, Any]
    gemini_api_key: str

# Define node functions
def perception_node(state: DrivingState) -> Dict[str, Any]:
    agent = PerceptionAgent()
    # If path is provided, process file, else process array
    input_data = state["frame_path"] if state.get("frame_path") else state["image_data"]
    processed_img, results = agent.process_frame(input_data)
    return {
        "processed_image": processed_img,
        "perception_results": results
    }

def environment_node(state: DrivingState) -> Dict[str, Any]:
    agent = EnvironmentAgent()
    # Modifies perception_results detections risk levels in-place
    perception_results = state["perception_results"].copy()
    context = agent.analyze_environment(perception_results)
    return {
        "environment_context": context,
        "perception_results": perception_results
    }

def decision_node(state: DrivingState) -> Dict[str, Any]:
    agent = DecisionAgent()
    decision = agent.make_decision(state["environment_context"])
    return {
        "proposed_decision": decision
    }

def reasoning_node(state: DrivingState) -> Dict[str, Any]:
    api_key = state.get("gemini_api_key", None)
    agent = ReasoningAgent(api_key=api_key)
    explanation = agent.generate_explanation(
        state["perception_results"],
        state["environment_context"],
        state["proposed_decision"]
    )
    return {
        "reasoning": explanation
    }

def safety_node(state: DrivingState) -> Dict[str, Any]:
    agent = SafetyAgent()
    validated = agent.validate_action(
        state["perception_results"],
        state["environment_context"],
        state["proposed_decision"]
    )
    return {
        "validated_decision": validated
    }

# Build LangGraph Workflow
def build_agent_graph():
    workflow = StateGraph(DrivingState)
    
    # Add Nodes
    workflow.add_node("perception", perception_node)
    workflow.add_node("environment", environment_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("safety", safety_node)
    
    # Define Edges (linear sequential pipeline representing agent orchestration)
    workflow.set_entry_point("perception")
    workflow.add_edge("perception", "environment")
    workflow.add_edge("environment", "decision")
    workflow.add_edge("decision", "reasoning")
    workflow.add_edge("reasoning", "safety")
    workflow.add_edge("safety", END)
    
    # Compile
    return workflow.compile()
