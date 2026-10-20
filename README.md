# Production Heart Disease Classification System (MLOps End-to-End Pipeline)

**Author:** Sumukh M Lohit (2025AF05011)  
**Program:** BITS Pilani WILP M.Tech in Artificial Intelligence & Machine Learning  
**Course:** Machine Learning Operations (MLOps) — Assignment 01

---

## 1. Project Overview
This repository contains a scalable, reproducible, and production-ready machine learning pipeline engineered to predict the risk of coronary heart disease based on clinical diagnostic metrics from the UCI Machine Learning Repository (Cleveland database). 

The system implements modern MLOps best practices, covering automated data cleaning, exploratory data analysis, hyperparameter tuning, experiment tracking, model artifact serialization, automated unit testing, containerization, Kubernetes orchestration, and real-time telemetry logging].

---

## 2. Repository Architecture & Folder Hierarchy
```text
heart-disease-mlops/
│
├── data/
│   └── raw/
│       └── heart.csv              # Cleaned and binarized dataset (297 records)
│
├── docs/
│   └── screenshots/               # EDA visual plots and system execution proofs
│
├── k8s/
│   └── deployment.yaml            # Kubernetes Deployment, Resource Limits, Probes, & LoadBalancer Service
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD pipeline workflow (Lint, Test)
│
├── src/
│   └── app.py                     # Production FastAPI web service, Pydantic schemas, & Prometheus instrumentation
│
├── tests/
│   └── test_api.py                # Pytest unit testing suite (data, artifacts, and inference validation)
│
├── mlruns/                        # MLflow experiment tracking logs and metrics repository
│
├── Dockerfile                     # Multi-stage lightweight container build spec (python:3.10-slim)
├── requirements.txt               # Standardized runtime dependency manifest
├── model.pkl                      # Serialized primary production model artifact (Tuned Random Forest)
├── scaler.pkl                     # Serialized StandardScaler transformer artifact
└── README.md                      # Project documentation
