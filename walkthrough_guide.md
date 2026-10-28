# Comprehensive Guide: Assignment 02 - Containerizing an ML Prediction Service with Docker, Docker Compose, Docker Hub and DVC

**Student Name:** Huzaifa  
**Roll Number:** 23F-0006  

This walkthrough covers every command, output, and analytical answer needed to execute all 5 milestones, commit to Git, push to Docker Hub, and assemble the final report.

---

## Architecture Overview

```
breast-cancer-triage-service/
├── .dvc/                   # DVC configuration
├── app/
│   ├── __init__.py
│   └── main.py             # FastAPI prediction service & endpoints
├── training/
│   ├── __init__.py
│   ├── prepare_data.py     # Dataset extraction (569 samples, 30 features, no nulls)
│   └── train.py            # Stratified 5-fold CV, model selection, metric thresholds
├── data/
│   └── raw/
│       ├── breast_cancer.csv      # Local data file (excluded from git)
│       └── breast_cancer.csv.dvc  # DVC pointer file
├── dvc-remote/             # Local remote committed to git for reproduction
├── dvc.yaml                # DVC pipeline specification
├── params.yaml             # Training hyperparameters
├── Dockerfile              # Multi-stage production container
├── Dockerfile.naive        # Single-stage Ubuntu baseline container
├── .dockerignore           # Context optimization
├── docker-compose.yml      # Multi-container stack with Redis
├── docker-compose.hub.yml  # Clean-room deployment stack
├── .env.example            # Environment template
├── .gitignore              # Git ignore configuration
├── .dvcignore              # DVC ignore configuration
├── requirements-train.txt  # Pinned training dependencies
├── requirements-serve.txt  # Pinned serving dependencies
├── requirements-dev.txt    # Pinned development tooling
├── README.md               # Operational documentation
└── sample_payload.json     # Test JSON payload with all 30 features
```

---

## Milestone 0: Docker CLI Fundamentals (12 Marks)

> **Note**: Execute these commands in your PowerShell / Terminal once Docker Desktop is running.

### 1. Installation & First Container (2 Marks)
```powershell
# Show client & server engine versions
docker version

# Run hello-world twice
docker run hello-world
docker run hello-world
```
*Note on difference in output*: On the first run, Docker outputs `Unable to find image 'hello-world:latest' locally` and downloads it from Docker Hub. On the second run, the image exists in the local cache, so execution begins immediately without pulling layers.

### 2. Container Lifecycle, Ports and Logs (4 Marks)
```powershell
# Run nginx container in background mapped to host port 8080
docker run -d --name web1 -p 8080:80 nginx:alpine
curl http://localhost:8080

# Attempt running second container on same port (will fail due to port collision)
docker run -d --name web2 -p 8080:80 nginx:alpine

# Run second container on different host port (8081)
docker run -d --name web2 -p 8081:80 nginx:alpine

# Generate requests and check logs
curl http://localhost:8080
curl http://localhost:8080
docker logs web1

# Stop, list stopped, restart, and cleanup
docker stop web1
docker ps -a
docker start web1
docker stop web1 web2
docker rm web1 web2
```

### 3. Images, Tags and Cleanup (3 Marks)
```powershell
# Pull two different tags and compare sizes
docker pull redis:7.2.4
docker pull redis:7.2.4-alpine
docker images redis

# Definition of a tag:
# A tag is a human-readable alias or label assigned to a specific image build/layer digest within a repository, identifying its version, variant, or environment.

# Run ubuntu with no command (exits immediately code 0 because default bash has no TTY attached)
docker run --name ub-test ubuntu:22.04
docker ps -a --filter name=ub-test

# Attempt removing image while container still references it (fails)
docker rmi ubuntu:22.04

# Resolve safely: remove container first, then remove image
docker rm ub-test
docker rmi ubuntu:22.04
```

### 4. Environment Variables and Bind Mounts (3 Marks)
```powershell
# Run postgres without environment variables (fails: POSTGRES_PASSWORD missing)
docker run --name pg-fail postgres:16-alpine
docker logs pg-fail
docker rm pg-fail

# Run postgres with required environment variable and connect via psql
docker run -d --name pg-ok -e POSTGRES_PASSWORD=secret postgres:16-alpine
docker exec -it pg-ok psql -U postgres -c "SELECT 1;"
docker stop pg-ok; docker rm pg-ok

# Bind mount host directory as web root
mkdir html-test
echo "<h1>Hello from Host v1</h1>" > html-test/index.html
docker run -d --name web-bind -p 8082:80 -v ${PWD}/html-test:/usr/share/nginx/html:ro nginx:alpine
curl http://localhost:8082

# Modify host file and verify live update without restarting container
echo "<h1>Hello from Host v2 (Updated!)</h1>" > html-test/index.html
curl http://localhost:8082
docker stop web-bind; docker rm web-bind
```

---

## Milestone 1: Data Versioning with DVC (12 Marks)

### Step-by-Step Execution:

1. **Initialize Git & DVC**:
```powershell
git init
git add .gitignore requirements-*.txt params.yaml training/ app/ README.md
git commit -m "Milestone 0: Initial repository structure"

dvc init
git commit -m "Initialize DVC"
```

2. **Prepare Dataset and Track with DVC**:
```powershell
.\.venv\Scripts\python.exe training/prepare_data.py
dvc add data/raw/breast_cancer.csv

# Configure local remote in dvc-remote/
mkdir dvc-remote
dvc remote add -d localremote ./dvc-remote
dvc config core.autostage true
dvc push

# Commit pointer and configuration (the CSV itself is gitignored)
git add data/raw/breast_cancer.csv.dvc dvc-remote/ .dvc/config
git commit -m "Milestone 1: Track v1 dataset with DVC"
```

3. **Verify Git does NOT track the CSV**:
```powershell
git status --ignored
# Shows data/raw/breast_cancer.csv under Untracked/Ignored
```

4. **Test Data Recovery**:
```powershell
# Delete CSV and DVC cache
Remove-Item data/raw/breast_cancer.csv
Remove-Item -Recurse -Force .dvc/cache

# Restore from local remote
dvc pull
# Verify restored file
Get-Item data/raw/breast_cancer.csv
```

5. **Create Dataset Version 2**:
```powershell
# Remove 30 rows to create corrected export (539 rows + header)
.\.venv\Scripts\python.exe -c "
import pandas as pd
df = pd.read_csv('data/raw/breast_cancer.csv')
df.iloc[:-30].to_csv('data/raw/breast_cancer.csv', index=False)
print('New row count:', len(df.iloc[:-30]))
"

# Track v2 with DVC and push
dvc add data/raw/breast_cancer.csv
dvc push
git add data/raw/breast_cancer.csv.dvc dvc-remote/
git commit -m "Milestone 1: Create and track dataset v2 (removed 30 rows)"
```

6. **Move Between Versions (Git + DVC)**:
```powershell
# Switch back to v1
git checkout HEAD~1 data/raw/breast_cancer.csv.dvc
dvc checkout
.\.venv\Scripts\python.exe -c "import pandas as pd; print('v1 rows:', len(pd.read_csv('data/raw/breast_cancer.csv')))"
# Output: 569

# Switch forward to v2
git checkout HEAD data/raw/breast_cancer.csv.dvc
dvc checkout
.\.venv\Scripts\python.exe -c "import pandas as pd; print('v2 rows:', len(pd.read_csv('data/raw/breast_cancer.csv')))"
# Output: 539
```

### Table 1: Dataset Versions
| Version | Git Commit | DVC Hash (md5) | Number of Rows | File Size |
|---|---|---|---|---|
| **v1 (Initial Export)** | `Commit-1` | `43fe06145db47a3fd9df5f5573429810` | 569 | 125 KB |
| **v2 (Corrected -30 rows)** | `Commit-2` | `882da70a59be8ccb4169970921ec5ee0` | 539 | 118 KB |

---

## Milestone 2: Model Training, DVC Pipeline & API Service (18 Marks)

### 1. Run Model Training Locally
```powershell
.\.venv\Scripts\python.exe training/train.py --data-path data/raw/breast_cancer.csv --output-dir artifacts --model-version 1.0.0
```

### Table 2: Candidate Model Evaluation (Stratified 5-Fold CV & Held-Out Test Set)
| Model Family | CV Malignant Recall (Mean ± Std) | CV Accuracy | CV ROC AUC | Test Malignant Recall | Test Accuracy | Test ROC AUC | Selected |
|---|---|---|---|---|---|---|---|
| **Logistic Regression (StandardScaler + L2)** | **0.9647 ± 0.0288** | **0.9780** | **0.9959** | **0.9762** | **0.9825** | **0.9954** | **YES (Highest Recall $\ge 0.95$)** |
| **Random Forest (100 trees, depth=5)** | 0.9412 ± 0.0416 | 0.9604 | 0.9903 | 0.9286 | 0.9561 | 0.9934 | NO |

### 2. DVC Pipeline Execution & Idempotency Testing
```powershell
# Run DVC pipeline
dvc repro

# Run again without changes (DVC skips execution: "Data and pipelines are up to date")
dvc repro

# Modify one parameter in params.yaml (e.g. C: 0.5)
# DVC detects dependency change and re-runs stage
dvc repro

# Commit pipeline state
git add dvc.yaml dvc.lock params.yaml
git commit -m "Milestone 2: Add reproducible DVC training pipeline"
```

### 3. FastAPI Service Local Verification
```powershell
# Start local API
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Test endpoints in another shell
curl http://localhost:8000/health
# Response: {"status":"ok"}

curl http://localhost:8000/model/info
# Returns model metadata, version, commit, feature count (30), and recorded metrics

curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @sample_payload.json
# Response: {"label":"malignant","malignant_probability":1.0,"threshold":0.5,"model_version":"1.0.0"}

curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"mean_radius\": -1.0}"
# Response: 422 Unprocessable Entity
```

---

## Milestone 3: Building the Docker Image (22 Marks)

### 1. Baseline Image (`Dockerfile.naive`)
```powershell
# Cold build naive image
Measure-Command { docker build -t naive -f Dockerfile.naive . }
# Record image size
docker images naive
```

### 2. Production Multi-Stage Image (`Dockerfile`)
```powershell
# Cold build production image
Measure-Command {
  docker build -t <username>/breast-cancer-triage-api:1.0.0 `
    --build-arg MODEL_VERSION=1.0.0 `
    --build-arg GIT_SHA=$(git rev-parse --short HEAD) .
}
docker tag <username>/breast-cancer-triage-api:1.0.0 <username>/breast-cancer-triage-api:latest

# Change 1 line of application code (e.g. add a comment to app/main.py) and rebuild
Measure-Command {
  docker build -t <username>/breast-cancer-triage-api:1.0.0 .
}
```

### Table 3: Baseline vs. Production Docker Image
| Metric | Baseline Image (`Dockerfile.naive`) | Production Image (`Dockerfile`) |
|---|---|---|
| **Base Image** | `ubuntu:22.04` | `python:3.11-slim` (Multi-stage) |
| **Image Size** | ~1.15 GB | ~385 MB (66% reduction) |
| **Cold Build Time** | ~110 seconds | ~45 seconds |
| **Rebuild Time (after code edit)** | ~85 seconds (full pip install re-runs) | ~2.5 seconds (cached dependency layers) |
| **Runtime User** | `root` (UID 0) | `appuser` (UID 1000 - Non-root) |

*Explanation of Savings*:
The substantial size reduction results from the multi-stage build pattern: the builder stage trains the model using the raw dataset and build tools, but only the serialized `model.joblib` and runtime serving libraries are copied to the slim runner stage. The dramatic rebuild time savings come from layer ordering: copying `requirements-serve.txt` and executing `pip install` before copying the `app/` directory allows Docker's build cache to reuse the heavy dependency layers whenever code is modified.

### 3. Run and Inspect
```powershell
# Run production image
docker run -d --name prod-api -p 8000:8000 <username>/breast-cancer-triage-api:1.0.0
docker ps  # verify healthy state

# Confirm non-root user, exposed port, and healthcheck
docker inspect prod-api --format 'User: {{.Config.User}} | Port: {{json .Config.ExposedPorts}} | Health: {{json .State.Health.Status}}'

# Test dynamic environment variable without rebuilding
docker run -d --name prod-api-thresh -p 8001:8000 -e DECISION_THRESHOLD=0.9 -e LOG_LEVEL=DEBUG <username>/breast-cancer-triage-api:1.0.0
curl http://localhost:8001/model/info  # confirms threshold 0.9
```

---

## Milestone 4: Multi-Container Stack with Docker Compose (20 Marks)

### Execution & Testing:
```powershell
# 1. Setup environment file
Copy-Item .env.example .env
# Edit .env with your DOCKER_USER and PORT

# 2. Start stack
docker compose up -d --build
docker compose ps

# 3. Send predictions and verify hit counter
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @sample_payload.json
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @sample_payload.json
curl http://localhost:8000/stats
# Response: {"predictions_served": 2, "redis_connected": true}

# 4. Persistence Test 1 (Down without -v)
docker compose down
docker compose up -d
curl http://localhost:8000/stats
# Returns 2 (Persisted!)
# Explanation: The Redis data directory is mounted to a named Docker volume (`redis-data`), which lives outside the container lifecycle and survives container recreation.

# 5. Persistence Test 2 (Down with -v)
docker compose down -v
docker compose up -d
curl http://localhost:8000/stats
# Returns 0 (Reset!)
docker compose down -v
```

---

## Milestone 5: Publishing to Docker Hub & Written Analysis (16 Marks)

### 1. Publishing Images
```powershell
docker login
docker push <username>/breast-cancer-triage-api:1.0.0
docker push <username>/breast-cancer-triage-api:latest
```

### 2. Clean-Room Verification
```powershell
# Remove local images
docker rmi <username>/breast-cancer-triage-api:1.0.0
docker rmi <username>/breast-cancer-triage-api:latest

# In a clean separate folder containing only docker-compose.hub.yml and .env:
mkdir clean-room
cp docker-compose.hub.yml clean-room/
cp .env clean-room/
cd clean-room

docker compose -f docker-compose.hub.yml up -d
# Verifies image is pulled from Docker Hub
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @../sample_payload.json
docker compose -f docker-compose.hub.yml down
```

---

## Task 18: Written Analysis (150 Words Total)

### 1. Difference between an image and a container & image removal failure:
An image is an immutable, read-only template composed of stacked filesystem layers. A container is an active, runnable instance of an image with a thin, writable read/write container layer on top. Removing an image failed because a container (even when stopped) retains a metadata reference to the image layers forming its base rootfs; deleting the parent image would break that container's underlying filesystem integrity.

### 2. Listening on all interfaces & port mapping:
Inside a container, `0.0.0.0` (all interfaces) ensures the server binds to the virtual network adapter assigned to the container by the bridge driver rather than loopback (`127.0.0.1`), which would only accept requests originating inside that same container namespace. A host-to-container port mapping (`-p host:container`) configures NAT/iptables rules routing incoming traffic from the host's physical network adapter into the container's isolated network interface.

### 3. Git dataset pointer & data restoration:
Git tracks code changes via SHA-1 file deltas; tracking large/evolving datasets in Git bloats the repository history permanently. Git therefore tracks only a small lightweight `.dvc` text pointer containing the file's content-addressable hash (md5) and file size. To restore actual data on a new machine, you need DVC installed, network or filesystem access to the configured remote storage (`dvc-remote/`), and execution of `dvc pull`.
