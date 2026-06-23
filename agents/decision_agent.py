class DecisionAgent:
    def make_decision(self, environment_context):
        """
        Processes environment context and generates a candidate autonomous driving action
        based on prioritized safety rules.
        """
        ped_danger = environment_context.get("pedestrian_danger", False)
        closest_obstacle = environment_context.get("closest_obstacle")
        active_signs = environment_context.get("active_signs", [])
        lane_status = environment_context.get("lane_status", {})
        lane_warning = lane_status.get("warning", "Normal")
        
        proposed_action = "Maintain Speed"
        confidence = 0.95
        rationale = "Road ahead is clear. Maintaining current cruise speed."
        
        # Priority 1: Pedestrian danger in paths (must Stop immediately)
        if ped_danger:
            proposed_action = "Stop Vehicle"
            confidence = 0.98
            rationale = "Pedestrian detected in critical path. Safety protocol mandates immediate stop."
            return {"action": proposed_action, "confidence": confidence, "rationale": rationale}
            
        # Priority 2: Active Stop Signs or Red Lights close by
        for sign in active_signs:
            if sign["type"] == "stop sign" and sign["risk"] in ["Collision", "Warning"]:
                proposed_action = "Stop Vehicle"
                confidence = 0.96
                rationale = "Approaching Stop Sign. Decelerating to complete stop at junction."
                return {"action": proposed_action, "confidence": confidence, "rationale": rationale}
            elif sign["type"] == "no entry" and sign["risk"] in ["Collision", "Warning"]:
                proposed_action = "Stop Vehicle"
                confidence = 0.99
                rationale = "No Entry sign detected ahead. Halting vehicle immediately to prevent illegal lane entry."
                return {"action": proposed_action, "confidence": confidence, "rationale": rationale}
                
        # Priority 3: Immediate Obstacle Collision (Vehicles, debris)
        if closest_obstacle and closest_obstacle["risk_level"] == "Collision":
            # Decide whether to brake or change lanes
            # If lane departure is normal, we might change lanes if possible, but standard safety rule is to decelerate first
            proposed_action = "Slow Down"
            confidence = 0.92
            rationale = f"Obstacle ({closest_obstacle['class_name']}) is extremely close ({closest_obstacle['distance']}m). Applying active braking."
            return {"action": proposed_action, "confidence": confidence, "rationale": rationale}
            
        # Priority 4: Lane Departure Warning (Corrective actions)
        if "WARNING" in lane_warning:
            confidence = 0.85
            if "Left" in lane_warning:
                proposed_action = "Change Lane Right"
                rationale = "Vehicle drift detected on left lane line. Proposing drift correction to the right."
            else:
                proposed_action = "Change Lane Left"
                rationale = "Vehicle drift detected on right lane line. Proposing drift correction to the left."
            return {"action": proposed_action, "confidence": confidence, "rationale": rationale}
            
        # Priority 5: Warning Zone Obstacle (deceleration/following distance)
        if closest_obstacle and closest_obstacle["risk_level"] == "Warning":
            proposed_action = "Slow Down"
            confidence = 0.88
            rationale = f"Detected {closest_obstacle['class_name']} in warning zone ({closest_obstacle['distance']}m). Reducing speed to increase following distance."
            return {"action": proposed_action, "confidence": confidence, "rationale": rationale}
            
        # Priority 6: Yield Sign
        for sign in active_signs:
            if sign["type"] == "yield" and sign["risk"] == "Warning":
                proposed_action = "Slow Down"
                confidence = 0.90
                rationale = "Yield sign detected. Slowing down and scanning intersection for crossing traffic."
                return {"action": proposed_action, "confidence": confidence, "rationale": rationale}
            elif sign["type"] == "speed limit 50" and sign["risk"] == "Warning":
                proposed_action = "Maintain Speed"
                confidence = 0.92
                rationale = "Speed limit 50 km/h detected. Calibrating cruise control limit."
                return {"action": proposed_action, "confidence": confidence, "rationale": rationale}
                
        # Default: Cruise / Maintain Speed
        return {"action": proposed_action, "confidence": confidence, "rationale": rationale}
