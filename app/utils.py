import streamlit as st
import torch
from config.config import CHAMPION_MODEL_DIR
import mlflow

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource()
def load_model():
    model = mlflow.pytorch.load_model(str(CHAMPION_MODEL_DIR))
    model.to(DEVICE)
    model.eval()
    return model