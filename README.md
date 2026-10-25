# Breast Cancer Triage Service

**Student Name:** Huzaifa  
**Roll Number:** 23F-0006  

Screening support tool that classifies cell-nucleus measurements from fine-needle aspirate (FNA) samples as likely malignant or likely benign to prioritize pathology review.

## Architecture & Project Structure

```text
breast-cancer-triage-service/
├── .dvc/                   # DVC configuration and tracking
├── app/
│   ├── __init__.py
│   └── main.py             # FastAPI prediction service & endpoints
├── training/
│   ├── __init__.py
│   ├── prepare_data.py     # Dataset extraction & verification
│   └── train.py            # Model training, 5-fold CV, and artifact export
├── data/
│   └── raw/
│       ├── breast_cancer.csv      # Local data file (ignored by Git)
│       └── breast_cancer.csv.dvc  # DVC pointer tracking dataset
├── dvc-remote/             # Local DVC remote (tracked in Git for reproduction)
├── dvc.yaml                # DVC pipeline stages
├── dvc.lock                # DVC reproducible pipeline lock
├── params.yaml             # Hyperparameters and seed configurations
├── Dockerfile              # Production multi-stage Docker build
├── Dockerfile.naive        # Baseline single-stage Docker build (Ubuntu-based)
├── .dockerignore           # Context optimization ignore rules
├── docker-compose.yml      # Local multi-container stack (API + Redis)
├── docker-compose.hub.yml  # Clean-room deployment stack from Docker Hub
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
├── .dvcignore              # DVC ignore rules
├── requirements-train.txt  # Training dependencies
├── requirements-serve.txt  # Production serving dependencies
├── requirements-dev.txt    # Local development & DVC tooling
└── README.md               # Complete operational documentation
```

---

## 1. Environment Setup

Create and activate a fresh virtual environment:
```powershell
python -m venv .venv
# Activate in PowerShell:
.\.venv\Scripts\Activate.ps1
# Or in Command Prompt:
.\.venv\Scripts\activate.bat

# Install all development and serving dependencies
pip install -r requirements-dev.txt
```

---

## 2. Restoring and Tracking Data with DVC

The dataset is tracked with DVC. The local remote `dvc-remote/` is committed to Git so anyone can reproduce the project.

### Restore data from remote:
```bash
dvc pull
```

### Reproduce dataset extraction:
```bash
python training/prepare_data.py
```

### Track dataset with DVC:
```bash
dvc add data/raw/breast_cancer.csv
dvc push
git add data/raw/breast_cancer.csv.dvc dvc-remote/ .dvc/config
git commit -m "Milestone 1: Track dataset version with DVC"
```

---

## 3. Running the Training Pipeline

Run the DVC pipeline:
```bash
dvc repro
```
This executes `training/train.py`, performs a stratified 80/20 train-test split (seed 42), benchmarks Logistic Regression and Random Forest using 5-fold cross-validation, validates that the held-out malignant recall $\ge 0.95$, and outputs:
- `artifacts/model.joblib`: Serialized scikit-learn pipeline including scaler
- `artifacts/model_card.json`: Model metadata, metrics, and parameters
- `artifacts/metrics.json`: Evaluated metrics for DVC tracking

---

## 4. Running the API Service Locally

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Access Swagger UI interactive documentation:
`http://localhost:8000/docs`

Test the health endpoint:
```bash
curl http://localhost:8000/health
```

Sample prediction call:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d @sample_payload.json
```

---

## 5. Docker Packaging & Multi-Container Stack

### Build Baseline (Naive) Image:
```bash
docker build -t naive -f Dockerfile.naive .
```

### Build Production Multi-Stage Image:
```bash
docker build -t <dockerhub_username>/breast-cancer-triage-api:1.0.0 \
             --build-arg MODEL_VERSION=1.0.0 \
             --build-arg GIT_SHA=$(git rev-parse --short HEAD) .
```

### Run Multi-Container Stack (FastAPI + Redis):
```bash
# Copy and configure environment
cp .env.example .env

# Start stack with Docker Compose
docker compose up -d --build

# Check services and health status
docker compose ps

# Check prediction stats counted in Redis
curl http://localhost:8000/stats

# Stop and remove containers (preserving volume)
docker compose down

# Stop and remove containers and Redis volume
docker compose down -v
```

---

## 6. Docker Hub Clean-Room Verification

To verify that the service can run without local source files:
```bash
docker compose -f docker-compose.hub.yml up -d
curl http://localhost:8000/health
curl http://localhost:8000/model/info
docker compose -f docker-compose.hub.yml down
```
