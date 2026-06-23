import cv2
import numpy as np
from configs.settings import LANE_ROI_VERTICES, LANE_DEPARTURE_THRESHOLD

def get_roi_mask(img, vertices):
    """Applies a region of interest mask based on polygon vertices."""
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)

def detect_lanes(img):
    """
    Performs lane detection on a street image.
    Returns:
        lane_overlay: Image with lane drawing
        departure_warning: String indicating warning or "None"
        lane_metrics: Dictionary containing lane info
    """
    height, width = img.shape[:2]
    
    # 1. Convert to grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # 3. Region of Interest Mask
    roi_vertices = np.array([[
        (int(v[0] * width), int(v[1] * height)) for v in LANE_ROI_VERTICES
    ]], dtype=np.int32)
    masked_edges = get_roi_mask(edges, roi_vertices)
    
    # 4. Hough Line Transform
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=20, 
        minLineLength=30, 
        maxLineGap=150
    )
    
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue  # Skip vertical lines
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                # Filter out near-horizontal lines
                if abs(slope) < 0.3 or abs(slope) > 3.0:
                    continue
                
                # Standard camera view: Left lane has negative slope, Right lane has positive slope
                if slope < 0:
                    left_lines.append((slope, intercept))
                else:
                    right_lines.append((slope, intercept))
                    
    # 5. Average/Extrapolate Lines
    y_min = int(height * 0.6)
    y_max = height
    
    left_lane = None
    right_lane = None
    
    if left_lines:
        avg_slope, avg_intercept = np.mean(left_lines, axis=0)
        x_start = int((y_min - avg_intercept) / avg_slope)
        x_end = int((y_max - avg_intercept) / avg_slope)
        left_lane = (x_start, y_min, x_end, y_max)
        
    if right_lines:
        avg_slope, avg_intercept = np.mean(right_lines, axis=0)
        x_start = int((y_min - avg_intercept) / avg_slope)
        x_end = int((y_max - avg_intercept) / avg_slope)
        right_lane = (x_start, y_min, x_end, y_max)
        
    # 6. Create Visual Overlay & Compute Departure Metrics
    overlay = img.copy()
    lane_metrics = {"left_detected": left_lane is not None, "right_detected": right_lane is not None}
    departure_warning = "Normal"
    
    if left_lane is not None and right_lane is not None:
        # Draw lane polygon (green shaded area)
        pts = np.array([
            [left_lane[0], left_lane[1]],
            [left_lane[2], left_lane[3]],
            [right_lane[2], right_lane[3]],
            [right_lane[0], right_lane[1]]
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 180, 0))
        
        # Draw lane boundary lines
        cv2.line(overlay, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (0, 255, 0), 4)
        cv2.line(overlay, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 255, 0), 4)
        
        # Calculate lane center vs vehicle center
        lane_center = (left_lane[2] + right_lane[2]) / 2.0
        vehicle_center = width / 2.0
        deviation = (vehicle_center - lane_center) / width  # Normalized
        lane_metrics["deviation"] = deviation
        
        if deviation > LANE_DEPARTURE_THRESHOLD:
            departure_warning = "WARNING: Lane Departure Right!"
        elif deviation < -LANE_DEPARTURE_THRESHOLD:
            departure_warning = "WARNING: Lane Departure Left!"
    else:
        # Draw ROI boundaries for visual feedback if lane not detected
        cv2.polylines(overlay, [roi_vertices], isClosed=True, color=(0, 255, 255), thickness=2)
        if left_lane is not None:
            cv2.line(overlay, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (0, 255, 0), 3)
        if right_lane is not None:
            cv2.line(overlay, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 255, 0), 3)
        lane_metrics["deviation"] = 0.0
        
    # Alpha blend overlay for premium transparent green lane
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    return img, departure_warning, lane_metrics

def draw_perception_overlay(img, detections):
    """
    Draws bounding boxes, labels, distance and hazard markers on an image.
    detections: list of dicts with keys (bbox, class_name, confidence, distance, risk_level)
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["class_name"].capitalize()
        dist = det["distance"]
        risk = det["risk_level"]
        
        # Determine color scheme based on Risk Level
        if risk == "Collision":
            color = (0, 0, 255)       # Bright Red
            label_text = f"CRITICAL: {label} ({dist:.1f}m)"
        elif risk == "Warning":
            color = (0, 140, 255)     # Warm Orange
            label_text = f"WARNING: {label} ({dist:.1f}m)"
        else:
            color = (255, 180, 0)     # Neon Cyan / Slate Blue
            label_text = f"{label} ({dist:.1f}m)"
            
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Create premium label tag
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        
        text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
        tx1, ty1 = x1, y1 - 4
        tx2, ty2 = x1 + text_size[0] + 6, y1 - text_size[1] - 8
        
        # Draw filled background for label text
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), color, cv2.FILLED)
        
        # Draw white label text
        cv2.putText(img, label_text, (x1 + 3, y1 - 6), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    return img
