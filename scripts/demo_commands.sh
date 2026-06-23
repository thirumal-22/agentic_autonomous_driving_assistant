#!/usr/bin/env bash
# Demo script: start the Streamlit dashboard and open on default port

python3 -m venv .venv_demo
source .venv_demo/bin/activate
pip install -r requirements.txt
streamlit run app.py

# Note: for headless servers, set BROWSER env or use port forwarding.
