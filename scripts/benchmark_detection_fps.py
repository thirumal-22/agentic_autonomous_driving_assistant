#!/usr/bin/env python3
"""Benchmark detection FPS using YOLOv8 if available, else fallback to OpenCV ops."""
import time
import os
import sys
import cv2

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, 'yolov8n.pt')
SAMPLES_DIR = os.path.join(ROOT, 'data', 'samples')

IMG_PATHS = []
if os.path.isdir(SAMPLES_DIR):
    for f in os.listdir(SAMPLES_DIR):
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            IMG_PATHS.append(os.path.join(SAMPLES_DIR, f))

if not IMG_PATHS:
    # create a dummy image
    import numpy as np
    img = (np.random.rand(640,480,3)*255).astype('uint8')
    IMG_PATHS = [None]
else:
    img = None


def benchmark_opencv(iterations=50):
    print('Running OpenCV fallback benchmark...')
    if IMG_PATHS[0] is None:
        frame = img
    else:
        frame = cv2.imread(IMG_PATHS[0])
    # warmup
    for _ in range(5):
        _ = cv2.GaussianBlur(frame, (5,5), 0)
    start = time.time()
    for _ in range(iterations):
        _ = cv2.GaussianBlur(frame, (5,5), 0)
        _ = cv2.resize(frame, (320,240))
    elapsed = time.time()-start
    fps = iterations/elapsed if elapsed>0 else 0
    print(f'OpenCV ops: {fps:.1f} FPS over {iterations} iterations')
    return fps


def benchmark_yolo(iterations=20):
    try:
        from ultralytics import YOLO
    except Exception as e:
        print('ultralytics not available:', e)
        return None
    if not os.path.exists(MODEL_PATH):
        print('Model file not found at', MODEL_PATH)
        return None
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print('Failed to load YOLO model:', e)
        return None
    print('Running YOLOv8 benchmark...')
    if IMG_PATHS[0] is None:
        import numpy as np
        frame = img
    else:
        frame = cv2.imread(IMG_PATHS[0])[:,:,::-1]
    # warmup
    model.predict(frame)
    start = time.time()
    for _ in range(iterations):
        _ = model.predict(frame, verbose=False)
    elapsed = time.time()-start
    fps = iterations/elapsed if elapsed>0 else 0
    print(f'YOLOv8: {fps:.2f} FPS over {iterations} iterations')
    return fps


if __name__ == '__main__':
    yfps = benchmark_yolo()
    if yfps is None:
        benchmark_opencv()
    else:
        print('YOLO benchmark completed')
