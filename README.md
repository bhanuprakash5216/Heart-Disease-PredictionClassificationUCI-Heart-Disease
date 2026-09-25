# HeartAI

HeartAI is an educational full-stack machine-learning demo for exploring heart-disease risk classification with the re-processed Cleveland UCI dataset published through the [Kaggle dataset link](https://www.kaggle.com/code/jonbown/heart-disease-blood-glucose-cholesterol/data). It is **not a medical diagnostic system** and must not be used for clinical decisions.

## Stack

- Next.js, React, TypeScript, Tailwind CSS, Recharts
- FastAPI, scikit-learn, pandas, Joblib
- UCI Heart Disease / Cleveland data

## Run locally

### 1. Train the model

```bash
cd backend
python -m venv .venv
.venv\\Scripts\\activate       # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
python train_model.py --data ../dataset/heart_disease.csv
uvicorn main:app --reload --port 8000
```

The trainer downloads the Kaggle Cleveland UCI ZIP when the CSV is missing or empty, extracts `heart_cleveland_upload.csv`, normalizes its `condition` label to `target`, cleans missing values, fits preprocessing only on the training split, evaluates five classifiers, and writes `backend/models/model.joblib` plus `backend/models/metrics.json`.

### 2. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. Copy `.env.example` to `frontend/.env.local` if the API runs somewhere other than `http://localhost:8000`.

## Dataset

Download `heart_cleveland_upload.csv` from the Kaggle source and place it at `dataset/heart_disease.csv` (or let `train_model.py` download it automatically). The Kaggle file uses `condition` as its label; the trainer normalizes that to `target`. The expected columns are:

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, condition`

The Kaggle notebook identifies the underlying source as the UCI Heart Disease dataset and links the re-processed Cleveland file used here.

## API

- `GET /health`
- `GET /model-metrics`
- `GET /analytics`
- `POST /predict`

Swagger docs are available at http://localhost:8000/docs.

## Safety

Predictions are statistical outputs from a demonstration model. They can be wrong, are not a diagnosis, and should never replace a qualified healthcare professional.
