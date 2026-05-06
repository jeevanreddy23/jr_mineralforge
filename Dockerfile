FROM python:3.11-slim

LABEL description="MineralForge blast vibration risk predictor"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

EXPOSE 8000
EXPOSE 8501

CMD ["python", "train_pipeline.py"]
