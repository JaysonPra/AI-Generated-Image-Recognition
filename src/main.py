from fastapi import FastAPI, File, UploadFile
import torch
import mlflow
from mlflow.tracking import MlflowClient
from PIL import Image
import io
import time
import torch.nn.functional as F

from src.data.transforms import get_transformation
from src.data.dataset import ImageDataset
from src.data.dataloader import get_dataloader

app = FastAPI()
model = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "AI-Image-Classifier"

@app.on_event("startup")
def _load_model():
    """Loads the champion model
    """
    global model
    model_uri = f"models:/{MODEL_NAME}@champion"
    try:
        model = mlflow.pytorch.load_model(model_uri)
        model.to(DEVICE)
        model.eval()
        print("Model Loaded Successfully!")
    except Exception as e:
        print(f"Model Failed To Load: {e}")

def promote_to_champion(run_id:str):
    """Registers and promotes a model to champion, and loads it in memory

    Args:
        run_id (str): _description_

    Returns:
        _type_: _description_
    """
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=mv.version
    )

    _load_model()

    return mv.version

@app.post("/promote/{run_id}")
def promote_endpoint(run_id: str):
    """Endpoint for promoting a model to champion

    Args:
        run_id (str): MLFlow experiment's run_id

    Returns:
        dict: Status and Message
    """
    try:
        version = promote_to_champion(run_id)
        return {
            "status": "success",
            "message": "Model aliased successfully",
            "new_champion_version": version
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Gets an image and predicts whether it is real or AI generated.

    Args:
        file (UploadFile, optional): Image uploaded. Defaults to File(...).

    Returns:
        dict: Prediction and Confidence
    """
    start_time = time.perf_counter()

    content = await file.read()
    labels = {0: "Real", 1: "AI"}
    image = Image.open(io.BytesIO(content)).convert('RGB')

    transforms = get_transformation(config=None, is_training=False)
    input_tensor = transforms(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)

        confidence, class_id = torch.max(probabilities, dim=1)

    end_time = time.perf_counter()
    latency = end_time - start_time

    return {
        "prediction": labels[class_id.item()],
        "confidence": round(confidence.item(), 4),
        "latency_seconds": round(latency, 4)
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}