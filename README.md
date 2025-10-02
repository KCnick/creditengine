# Credit Scoring Engine

This project implements a credit scoring engine with:

- **Training** (XGBoost + preprocessing pipeline + MLflow logging)
- **Serving** (FastAPI REST API for predictions)
- **Monitoring** (drift detection with Evidently)
- **Deployment** (Docker, Kubernetes manifests)
- **CI/CD** (GitHub Actions)

## Quickstart
1. Train:
```bash
docker build -t credit-train -f Dockerfile.train .
docker run credit-train
