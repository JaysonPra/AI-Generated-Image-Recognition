import requests
import time
import streamlit as st

def wait_for_api(timeout=120, interval=2):
    start_time = time.time()

    with st.spinner("Waiting for inference API to be ready..."):
        while time.time() - start_time < timeout:
            try:
                health = requests.get("http://localhost:8000/health", timeout=3)
                if health.ok and health.json().get("model_loaded"):
                    return True
            except requests.ConnectionError:
                pass

            time.sleep(interval)

    return False

    