# JR MineralForge – Docker Configuration
# Team JR – Advancing Mineral Discovery with Robust AI & Open Australian Geodata

FROM python:3.11-slim

LABEL maintainer="Team JR <teamjr@example.com>"
LABEL description="JR MineralForge – IOCG Mineral Prospectivity AI Bot"
LABEL version="1.0.0"

# System dependencies for geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libspatialindex-dev \
    libgl1-mesa-glx \
    libgomp1 \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/vector_store reports models logs mlflow_runs

# Environment variables (override with docker-compose or -e flags)
ENV LLM_PROVIDER=ollama
ENV LLM_MODEL=llama3.1:8b
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV LOG_LEVEL=INFO
ENV PYTHONPATH=/app

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default: launch Gradio UI
CMD ["python", "app.py"]
