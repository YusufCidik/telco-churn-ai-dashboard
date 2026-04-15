# Telco Churn Prediction & AI Analytics Dashboard (Model Source Code In Docs)

Professional portfolio project that combines **churn prediction**, **explainable AI**, and an interactive **retention intelligence dashboard** for telecom customers.

## Project Goal

This project predicts customer churn risk and turns model output into actionable business insights:
- churn probability per customer
- explainability via SHAP-style top feature impacts
- AI-driven commentary for account teams
- dynamic retention offer generation by customer value

## Tech Stack

- **Backend:** FastAPI, scikit-learn ecosystem, XGBoost/CatBoost/LightGBM compatibility
- **Model Explainability:** SHAP (with robust fallback logic for ensemble models)
- **Frontend:** Next.js (App Router), Tailwind CSS, Recharts
- **Model Artifacts:** `model.pkl`, `scaler.pkl`, `feature_columns.pkl`

## Core Features

- **Explainable AI:** Top-5 feature impact per customer with direction (risk up/down)
- **Risk Segmentation:** Threshold-aware segments (`Düşük Risk`, `Orta Risk`, `KRİTİK`)
- **Dynamic Retention Engine:** ARPU-aware coupon recommendations
- **Visual Analytics:** Distribution histogram, trend cards, and customer-level insights

## Repository Structure

```text
.
├─ backend/
│  ├─ app/
│  │  ├─ main.py
│  │  └─ models/
│  │     ├─ model.pkl
│  │     ├─ scaler.pkl
│  │     └─ feature_columns.pkl
│  └─ requirements.txt
├─ frontend/
│  ├─ src/app/
│  ├─ .env.local.example
│  └─ package.json
└─ test_customers.csv
```

## Setup

### 1) Backend Setup (FastAPI)

```bash
cd backend
py -3 -m pip install -r requirements.txt
py -3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8080
```

Optional CORS origins:

```bash
set ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### 2) Frontend Setup (Next.js)

```bash
cd frontend
copy .env.local.example .env.local
npm install
npm run dev
```

Open: `http://localhost:3000`

## Model Artifacts (`.pkl`) Note

Large binary model files are usually excluded from Git repositories.  
For this portfolio version, artifacts are placed under `backend/app/models/`.

If you prefer not to commit them:
1. remove `.pkl` files from git tracking
2. upload artifacts to secure storage
3. document download instructions in deployment scripts

## API Endpoints

- `GET /health` - service health check
- `POST /api/analyze` - batch churn prediction + explainability + commentary
- `POST /api/coupon` - value-aware retention offer generation

## Screenshots (Placeholders)

Add your screenshots under `docs/screenshots/` and update links below:

![Dashboard Overview](docs/screenshots/dashboard-overview.png)
![Customer Explainability Panel](docs/screenshots/customer-explainability.png)
![Churn Distribution Analytics](docs/screenshots/churn-distribution.png)
