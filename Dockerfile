FROM python:3.10-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src /app/src

# Copy trained model
COPY model.joblib /app/model.joblib
COPY data/credit_train.parquet /app/data/credit_train.parquet

EXPOSE 8080

# Run FastAPI from the src.serving.app module
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
