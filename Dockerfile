FROM python:3.11-slim

WORKDIR /app

# Install the runtime library required by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Copy application and trained model
COPY src/serving/ src/serving/
COPY models/xgboost_baseline.json models/xgboost_baseline.json

# Expose the API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]