import os
import json

# Try new SDK first, fall back to deprecated SDK for compatibility
try:
    from google import genai
    _USE_NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai_legacy
        _USE_NEW_SDK = False
    except ImportError:
        _USE_NEW_SDK = None

from configs.settings import GEMINI_MODEL, DEFAULT_SYSTEM_INSTRUCTION

class ReasoningAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client_ready = False
        
        if self.api_key and _USE_NEW_SDK is not None:
            try:
                if _USE_NEW_SDK:
                    self.client = genai.Client(api_key=self.api_key)
                    self.client_ready = True
                else:
                    genai_legacy.configure(api_key=self.api_key)
                    self.model = genai_legacy.GenerativeModel(
                        model_name=GEMINI_MODEL,
                        generation_config={"response_mime_type": "application/json"},
                        system_instruction=DEFAULT_SYSTEM_INSTRUCTION
                    )
                    self.client_ready = True
            except Exception as e:
                print(f"Failed to configure Gemini client: {e}")
                self.client_ready = False

    def generate_explanation(self, perception_results, environment_context, proposed_decision):
        """
        Generates explainable AI output using Gemini.
        Falls back to rule-based template-based explanation if the client is not configured
        or fails.
        """
        action = proposed_decision.get("action")
        rationale = proposed_decision.get("rationale")
        confidence = proposed_decision.get("confidence")
        risk_score = environment_context.get("risk_score")
        
        # Prepare context representation for LLM
        detections_summary = []
        for det in perception_results.get("detections", []):
            detections_summary.append(
                f"- Detected {det['class_name']} at {det['distance']} meters ({det['risk_level']} Risk)"
            )
            
        context_str = f"""
        System State Context:
        - Candidate Decision Action: {action}
        - Initial Rationale: {rationale}
        - Confidence Score: {confidence}
        - Environment Risk Score: {risk_score}/10.0
        - Road Condition: {environment_context.get('road_condition')}
        - Lane Status: {environment_context.get('lane_status', {}).get('warning', 'Normal')}
        - Detected Objects:
        {chr(10).join(detections_summary) if detections_summary else 'No objects detected.'}
        """

        if self.client_ready:
            try:
                prompt = f"""
                Analyze the driving state and explain the recommended action.
                {context_str}
                
                Remember to output in strict JSON format with keys: explanation, safety_analysis, final_recommendation.
                """
                if _USE_NEW_SDK:
                    response = self.client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=DEFAULT_SYSTEM_INSTRUCTION + "\n\n" + prompt
                    )
                    res_data = json.loads(response.text.strip())
                else:
                    response = self.model.generate_content(prompt)
                    res_data = json.loads(response.text.strip())
                return {
                    "explanation": res_data.get("explanation", rationale),
                    "safety_analysis": res_data.get("safety_analysis", "Dynamic safety check passed."),
                    "final_recommendation": res_data.get("final_recommendation", f"Execute: {action}"),
                    "is_llm_generated": True
                }
            except Exception as e:
                # If Gemini fails, fallback to rule-based explanation
                print(f"Gemini generation error (falling back): {e}")
                
        # Rule-based fallback generator
        return self._generate_fallback_explanation(action, environment_context, rationale)

    def _generate_fallback_explanation(self, action, environment_context, rationale):
        """Generates structured local reasoning if Gemini is unavailable."""
        closest_obstacle = environment_context.get("closest_obstacle")
        lane_warning = environment_context.get("lane_status", {}).get("warning", "Normal")
        
        safety_analysis = "Standard safety margins are monitored."
        final_recommendation = f"Execute: {action}"
        
        if action == "Stop Vehicle":
            if environment_context.get("pedestrian_danger"):
                explanation = f"Pedestrian detected inside safety envelope ({closest_obstacle['distance'] if closest_obstacle else 'N/A'}m). Vehicle halting immediately to prevent a critical collision."
                safety_analysis = "Emergency safety constraint: Collision vulnerability at close range. Zero velocity required."
                final_recommendation = "Engage full hydraulic brakes, zero throttle, activate hazard warning lights."
            else:
                active_sign_names = [s["type"] for s in environment_context.get("active_signs", [])]
                if "stop sign" in active_sign_names:
                    explanation = "Stop sign detected ahead. Halting the vehicle to comply with road regulation and check for cross traffic."
                    safety_analysis = "Regulatory compliance constraint: Complete stop at stop line."
                    final_recommendation = "Gradual braking, stop at the designated line, restart scan sequence."
                else:
                    explanation = "Obstacle detected at critical distance. Initiating emergency stop to maintain vehicle clearance."
                    safety_analysis = "Immediate proximity hazard check: Clear corridor blocked."
                    final_recommendation = "Brake active, maintain current lane orientation."
                    
        elif action == "Slow Down":
            if closest_obstacle:
                explanation = f"Vehicle or object detected in the warning zone ({closest_obstacle['distance']}m). Reducing speed to increase following distance."
                safety_analysis = "Proactive distance buffering: Speed-dependent stopping distance rule."
                final_recommendation = "De-throttle, apply light braking, match lead vehicle velocity."
            else:
                explanation = "Approaching intersection or curve. Reducing speed for safe navigation."
                safety_analysis = "Dynamic stability margins: High lateral G-force risk avoidance."
                final_recommendation = "De-throttle and maintain lane alignment."
                
        elif "Change Lane" in action:
            explanation = f"Lane boundary departure detected ({lane_warning}). Correcting vehicle trajectory to center within lanes."
            safety_analysis = "Lateral lane tracking constraint: Departure warning triggers active steering correction."
            final_recommendation = f"Initiate minor steer correction towards center. Lateral deviation detected."
            
        else: # Maintain Speed
            explanation = "Road is clear, lane alignment is optimal, and no traffic signs are restricting cruise speed."
            safety_analysis = "Nominal operating envelope: All safety parameters within clear limits."
            final_recommendation = "Maintain standard cruise control throttle and active lane centering."
            
        return {
            "explanation": explanation,
            "safety_analysis": safety_analysis,
            "final_recommendation": final_recommendation,
            "is_llm_generated": False
        }
