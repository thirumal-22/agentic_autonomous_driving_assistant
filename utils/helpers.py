import os
import cv2
import numpy as np

def ensure_dir(path):
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)

def load_image(image_path):
    """Load image from path, handle errors."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    return img

def save_image(img, output_path):
    """Save image to path."""
    ensure_dir(os.path.dirname(output_path))
    cv2.imwrite(output_path, img)
