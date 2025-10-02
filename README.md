# Credit Scoring Engine

This project implements a credit scoring engine with:

- **Training** (XGBoost + preprocessing pipeline + MLflow logging)
- **Serving** (FastAPI REST API for predictions)
- **Monitoring** (drift detection with Evidently)

## Quickstart
1. Train:
```bash
docker build -t credit-train -f Dockerfile.train .
docker run credit-train
```

2. Test:
 ```bash
  docker build -t credit-serving -f Dockerfile .
  docker run -it --rm -p 5000:5000 -v $(pwd)/mlruns:/app/mlruns credit-serving
  curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{"age":30,"income":25000,"credit_lines":2,"delinquencies":0,"utilization":0.2,"months_active":24,"region":"Nairobi"}'
  ```
   Play around with deliquencies (i.e number of times application has failed to honor a payment), credit_lines, utilization for different results.
  

  
