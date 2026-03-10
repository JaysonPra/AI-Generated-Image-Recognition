import streamlit as st
import requests

st.markdown("## AI Image Recognition App\n _Created By_: **Jayson Pradhananga**")

uploaded_files = st.file_uploader(
    label="Choose images: ",
    type=["jpg", "jpeg", "webp", "png"],
    accept_multiple_files=True
)

for i, uploaded_file in enumerate(uploaded_files):
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    }

    response = requests.post("http://localhost:8000/predict/", files=files)
    result = response.json()

    st.markdown(f"File {i} is: **{result['prediction']}** with a confidence of **{result['confidence'] * 100}%**")