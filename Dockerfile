FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/main.py src/main.py
COPY src/data/transforms.py src/data/transforms.py
COPY src/data/dataloader.py src/data/dataloader.py
COPY config/config.py config/config.py

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]