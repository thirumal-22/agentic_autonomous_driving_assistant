#!/usr/bin/env bash
# Post-build fix for Streamlit Cloud: ensure only headless OpenCV is installed
set -e
python -m pip uninstall -y opencv-python opencv-python-headless || true
python -m pip install --no-cache-dir --force-reinstall opencv-python-headless==4.13.0.92
echo "Post-build: opencv-python-headless reinstalled"
