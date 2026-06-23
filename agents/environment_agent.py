from configs.settings import COLLISION_DISTANCE, WARNING_DISTANCE, SAFE_DISTANCE

class EnvironmentAgent:
    def analyze_environment(self, perception_results):
        """
        Analyzes the output of the perception agent to calculate risk levels,
        closest obstacles, and summarize road conditions.
        """
        detections = perception_results.get("detections", [])
        lane_warning = perception_results.get("lane_warning", "Normal")
        lane_metrics = perception_results.get("lane_metrics", {})
        
        analyzed_detections = []
        closest_distance = float('inf')
        closest_object = None
        pedestrian_danger = False
        active_signs = []
        
        # 1. Evaluate individual object risks and identify close objects
        for det in detections:
            det_copy = det.copy()
            dist = det_copy["distance"]
            cls_name = det_copy["class_name"]
            
            # Determine Risk Level based on distance
            if dist <= COLLISION_DISTANCE:
                det_copy["risk_level"] = "Collision"
            elif dist <= WARNING_DISTANCE:
                det_copy["risk_level"] = "Warning"
            else:
                det_copy["risk_level"] = "Safe"
                
            analyzed_detections.append(det_copy)
            
            # Check closest threat
            if dist < closest_distance:
                closest_distance = dist
                closest_object = det_copy
                
            # Check if pedestrian poses immediate danger
            if cls_name == "pedestrian" and dist <= WARNING_DISTANCE:
                pedestrian_danger = True
                
            # Log active traffic signs/signals
            if cls_name in ["stop sign", "speed limit 50", "yield", "no entry"]:
                active_signs.append({
                    "type": cls_name,
                    "distance": dist,
                    "risk": det_copy["risk_level"]
                })
                
        # 2. Calculate Road Condition Risk Score (0.0 to 10.0 scale)
        risk_score = 0.0
        if closest_object:
            # Pedestrians carry higher risk factors
            multiplier = 1.5 if closest_object["class_name"] == "pedestrian" else 1.0
            
            if closest_distance <= COLLISION_DISTANCE:
                # Close threat
                base = 8.0 + (COLLISION_DISTANCE - closest_distance) / COLLISION_DISTANCE * 2.0
                risk_score = min(10.0, base * multiplier)
            elif closest_distance <= WARNING_DISTANCE:
                # Warning zone
                base = 4.0 + (WARNING_DISTANCE - closest_distance) / (WARNING_DISTANCE - COLLISION_DISTANCE) * 4.0
                risk_score = base * multiplier
            else:
                # Safe zone
                risk_score = max(1.0, 4.0 - (closest_distance - WARNING_DISTANCE) / 20.0)
        else:
            # Safe highway
            risk_score = 1.0
            
        # Add risk if lane departure is active
        if "WARNING" in lane_warning:
            risk_score = min(10.0, risk_score + 2.0)
            
        # 3. Compile context
        environment_context = {
            "risk_score": round(risk_score, 1),
            "pedestrian_danger": pedestrian_danger,
            "closest_obstacle": closest_object,
            "active_signs": active_signs,
            "lane_status": {
                "warning": lane_warning,
                "deviation": lane_metrics.get("deviation", 0.0),
                "left_detected": lane_metrics.get("left_detected", False),
                "right_detected": lane_metrics.get("right_detected", False)
            },
            "road_condition": "Rainy/Wet" if "WARNING" in lane_warning else "Clear Highway"
        }
        
        # Write back risk levels to detections for overlay visualization
        perception_results["detections"] = analyzed_detections
        
        return environment_context
