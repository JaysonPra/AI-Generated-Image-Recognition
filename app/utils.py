import streamlit as st
import mlflow
import torch
from config.config import MODEL_NAME

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource()
def load_model():
    model_uri = f"models:/{MODEL_NAME}@champion"    
    model = mlflow.pytorch.load_model(model_uri)
    model.to(DEVICE)
    model.eval()
    return model