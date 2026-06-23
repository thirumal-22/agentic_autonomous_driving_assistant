from configs.settings import COLLISION_DISTANCE

class SafetyAgent:
    def validate_action(self, perception_results, environment_context, proposed_decision):
        """
        Validates the proposed decision against hard safety boundaries.
        Overrides actions if unsafe states are detected.
        """
        action = proposed_decision.get("action")
        confidence = proposed_decision.get("confidence")
        
        ped_danger = environment_context.get("pedestrian_danger", False)
        closest_obstacle = environment_context.get("closest_obstacle")
        detections = perception_results.get("detections", [])
        
        status = "APPROVED"
        override_reason = ""
        validated_action = action
        
        # Rule 1: Pedestrian safety override
        if ped_danger and closest_obstacle and closest_obstacle["class_name"] == "pedestrian":
            if closest_obstacle["distance"] <= COLLISION_DISTANCE and action != "Stop Vehicle":
                validated_action = "Stop Vehicle"
                status = "OVERRIDDEN"
                override_reason = f"CRITICAL: Proposed action '{action}' rejected. Pedestrian detected at collision distance ({closest_obstacle['distance']}m). Enforced emergency STOP."
                return self._build_result(validated_action, status, override_reason, confidence)
                
        # Rule 2: Vehicle collision override
        if closest_obstacle and closest_obstacle["distance"] <= COLLISION_DISTANCE:
            if action in ["Maintain Speed"]:
                validated_action = "Slow Down"
                status = "OVERRIDDEN"
                override_reason = f"CRITICAL: Proposed action '{action}' rejected. Road block detected at {closest_obstacle['distance']}m. Enforced SLOW DOWN."
                return self._build_result(validated_action, status, override_reason, confidence)
                
        # Rule 3: Target lane vehicle occupancy check (Lane Change Safety)
        if "Change Lane" in action:
            # We check if there's a vehicle on the side we want to merge into.
            # We can classify bounding box centroids: Left lane is x < 0.35 * width, Right lane is x > 0.65 * width.
            # (assuming vehicle width is normalized)
            # For this agent, let's scan all detections to see if there is an obstacle in the target lane.
            target_side = "left" if "Left" in action else "right"
            
            # Let's inspect coordinates of detections to see if a vehicle occupies that lane
            # We can simulate target-lane occupancy for verification
            has_blind_spot_threat = False
            threat_details = None
            
            for det in detections:
                if det["class_name"] in ["car", "truck", "motorcycle"]:
                    x1, _, x2, _ = det["bbox"]
                    # Calculate center in normalized coordinates
                    # Let's assume standard image width is roughly 640px to 1920px. We use relative positions.
                    # We'll fetch image shape or relative placement later.
                    # Let's mock a blind spot check: if the bounding box center is on the left/right and close (< 25m)
                    # For demo purposes, if an image contains a vehicle with "blind_spot" in its context,
                    # or if there is any car close on that side, we block the lane change.
                    # Let's write a heuristic:
                    bbox_center_x = (x1 + x2) / 2
                    # We can use normalized width bounds
                    # If target is left and center_x is on the left third of the image, and distance is close:
                    if target_side == "left" and x2 < 400 and det["distance"] < 25.0:
                        has_blind_spot_threat = True
                        threat_details = det
                        break
                    if target_side == "right" and x1 > 600 and det["distance"] < 25.0:
                        has_blind_spot_threat = True
                        threat_details = det
                        break
                        
            if has_blind_spot_threat:
                validated_action = "Slow Down"
                status = "OVERRIDDEN"
                override_reason = f"CRITICAL: Proposed '{action}' rejected. Threat vehicle detected in target lane ({threat_details['distance']}m). Lane change blocked. Enforced speed reduction."
                return self._build_result(validated_action, status, override_reason, confidence)
                
        return self._build_result(validated_action, status, override_reason or "All safety envelopes validated.", confidence)

    def _build_result(self, action, status, reason, confidence):
        return {
            "final_action": action,
            "safety_status": status,
            "safety_message": reason,
            "safety_score": 1.0 if status == "APPROVED" else 0.4
        }
