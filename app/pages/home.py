import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F

from src.data.transforms import get_transformation
from app.utils import load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.markdown("## AI Image Recognition App\n _Created By_: **Jayson Pradhananga**")

with st.spinner("Loading the model"):
    model = load_model()

uploaded_files = st.file_uploader(
    label="Choose images: ",
    type=["jpg", "jpeg", "webp", "png"],
    accept_multiple_files=True
)

for i, uploaded_file in enumerate(uploaded_files):
    image = Image.open(uploaded_file).convert("RGB")    
    transforms = get_transformation(config=None, is_training=False)
    input_tensor = transforms(image).unsqueeze(0).to(DEVICE)
    
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence, class_id = torch.max(probabilities, dim=1)
    
    labels = {0: "Real", 1: "AI"}
    st.markdown(f"Image {i+1}: **{labels[class_id.item()]}**. Confidence: **{round(confidence.item(), 4)*100}%**")