FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .
RUN pip install uv
RUN uv sync --no-dev --frozen

COPY src/main.py src/main.py
COPY src/data/transforms.py src/data/transforms.py
COPY src/data/dataloader.py src/data/dataloader.py
COPY config/config.py config/config.py

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]