import unittest
import numpy as np
from agents.environment_agent import EnvironmentAgent
from agents.decision_agent import DecisionAgent
from agents.safety_agent import SafetyAgent

class TestADASAgents(unittest.TestCase):
    def setUp(self):
        self.env_agent = EnvironmentAgent()
        self.dec_agent = DecisionAgent()
        self.saf_agent = SafetyAgent()

    def test_environment_risk_scoring(self):
        # Setup mock perception output with a close pedestrian
        perception_results = {
            "lane_warning": "Normal",
            "lane_metrics": {"deviation": 0.0},
            "detections": [
                {"bbox": [100, 100, 150, 250], "class_name": "pedestrian", "confidence": 0.9, "distance": 10.0, "risk_level": "Safe"}
            ]
        }
        context = self.env_agent.analyze_environment(perception_results)
        
        # Threat should be detected and risk score should be high
        self.assertTrue(context["pedestrian_danger"])
        self.assertGreater(context["risk_score"], 7.0)
        self.assertEqual(perception_results["detections"][0]["risk_level"], "Collision")

    def test_decision_making_rules(self):
        # Context with pedestrian danger
        context_danger = {
            "risk_score": 9.5,
            "pedestrian_danger": True,
            "closest_obstacle": {"class_name": "pedestrian", "distance": 10.0, "risk_level": "Collision"},
            "active_signs": [],
            "lane_status": {"warning": "Normal"}
        }
        decision = self.dec_agent.make_decision(context_danger)
        self.assertEqual(decision["action"], "Stop Vehicle")
        
        # Context with lane departure left
        context_drift = {
            "risk_score": 3.0,
            "pedestrian_danger": False,
            "closest_obstacle": None,
            "active_signs": [],
            "lane_status": {"warning": "WARNING: Lane Departure Left"}
        }
        decision = self.dec_agent.make_decision(context_drift)
        self.assertEqual(decision["action"], "Change Lane Right")

    def test_safety_override(self):
        # Situation: Proposed "Maintain Speed", but a pedestrian is in the collision zone
        perception_results = {
            "detections": [
                {"bbox": [100, 100, 150, 250], "class_name": "pedestrian", "confidence": 0.9, "distance": 10.0, "risk_level": "Collision"}
            ]
        }
        context = {
            "pedestrian_danger": True,
            "closest_obstacle": {"class_name": "pedestrian", "distance": 10.0, "risk_level": "Collision"}
        }
        proposed = {
            "action": "Maintain Speed",
            "confidence": 0.9
        }
        
        validated = self.saf_agent.validate_action(perception_results, context, proposed)
        
        # Safety agent must override "Maintain Speed" to "Stop Vehicle"
        self.assertEqual(validated["final_action"], "Stop Vehicle")
        self.assertEqual(validated["safety_status"], "OVERRIDDEN")
        self.assertIn("emergency STOP", validated["safety_message"])

if __name__ == '__main__':
    unittest.main()
