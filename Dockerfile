FROM python:3.11-slim

LABEL description="MineralForge edge-AI rock-burst risk prototype"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000
EXPOSE 8501

CMD ["python", "main.py", "--zone", "Stope 3", "--stress", "2.8"]
