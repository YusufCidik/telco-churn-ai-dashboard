from __future__ import annotations

import logging
import os
import secrets
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

THRESHOLD = 0.4790
APP_ROOT = Path(__file__).resolve().parent
MODELS_DIR = APP_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURES_PATH = MODELS_DIR / "feature_columns.pkl"
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


class AnalyzeRequest(BaseModel):
    """Request schema for batch customer churn analysis."""

    customers: list[dict[str, Any]] = Field(min_length=1, max_length=200)


class CouponRequest(BaseModel):
    """Request schema for generating retention coupon offers."""

    customer_id: str
    churn_probability: float
    risk_label: str
    monthly_charges: float = 0.0
    tenure: float = 0.0


def _allowed_origins() -> list[str]:
    """Read and sanitize allowed frontend origins from environment."""
    raw_origins = os.getenv("ALLOWED_ORIGINS", "")
    if not raw_origins.strip():
        return DEFAULT_ALLOWED_ORIGINS
    parsed = [origin.strip() for origin in raw_origins.split(",")]
    return [origin for origin in parsed if origin]


app = FastAPI(title="Telco Churn Analytics API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
logger = logging.getLogger("churn_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns: list[str] = joblib.load(FEATURES_PATH)
logger.info("Artifacts loaded. feature_count=%s", len(feature_columns))

try:
    import shap  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    shap = None

_cached_shap_explainer: Any | None = None


def _friendly_feature_name(feature: str) -> str:
    """Map encoded model feature names to human-readable labels."""
    if "_" not in feature:
        return feature
    prefix, suffix = feature.split("_", 1)
    return f"{prefix}: {suffix.replace('_', ' ')}"


def _risk_label(probability: float) -> str:
    """Convert churn probability into portfolio risk segments."""
    if probability < 0.30:
        return "Düşük Risk"
    if probability < THRESHOLD:
        return "Orta Risk"
    return "KRİTİK"


def _ensure_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create robust numeric and engineered features expected by model."""
    out = df.copy()
    out["tenure"] = pd.to_numeric(out.get("tenure", 0), errors="coerce").fillna(0)
    out["MonthlyCharges"] = pd.to_numeric(
        out.get("MonthlyCharges", 0), errors="coerce"
    ).fillna(0)
    out["TotalCharges"] = pd.to_numeric(out.get("TotalCharges", 0), errors="coerce").fillna(
        0
    )

    if "ChargePerTenure" not in out.columns:
        safe_tenure = out["tenure"].replace(0, 1)
        out["ChargePerTenure"] = (out["MonthlyCharges"] / safe_tenure).round(2)

    if "TenureGroup" not in out.columns:
        bins = [-1, 12, 24, 48, np.inf]
        labels = ["0-12", "13-24", "25-48", "48+"]
        out["TenureGroup"] = pd.cut(out["tenure"], bins=bins, labels=labels)
        out["TenureGroup"] = out["TenureGroup"].astype(str)

    return out


def _preprocess(customers: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate request rows and transform them into model-ready features."""
    raw_df = pd.DataFrame(customers)
    if raw_df.empty:
        raise HTTPException(status_code=400, detail="customers list cannot be empty")

    if "customerID" not in raw_df.columns:
        raw_df["customerID"] = [f"CUST-{idx + 1:04d}" for idx in range(len(raw_df))]

    engineered = _ensure_derived_features(raw_df)
    for col in ("Churn",):
        if col in engineered.columns:
            engineered = engineered.drop(columns=[col])

    encoded = pd.get_dummies(engineered, drop_first=False)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    scale_cols = [
        col for col in getattr(scaler, "feature_names_in_", []) if col in encoded.columns
    ]
    if scale_cols:
        scaled_values = scaler.transform(encoded[scale_cols].astype(float))
        for idx, col in enumerate(scale_cols):
            encoded[col] = scaled_values[:, idx]

    return raw_df, encoded


def _parse_uploaded_file(file_name: str, content: bytes) -> list[dict[str, Any]]:
    """Parse uploaded customer file into a list of records."""
    lower_name = file_name.lower()
    if lower_name.endswith(".csv"):
        df = pd.read_csv(BytesIO(content))
    elif lower_name.endswith(".json"):
        df = pd.read_json(BytesIO(content))
    elif lower_name.endswith(".xlsx") or lower_name.endswith(".xls"):
        df = pd.read_excel(BytesIO(content))
    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Allowed: .csv, .json, .xlsx, .xls",
        )

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return df.to_dict(orient="records")


def _compute_impacts(scaled_df: pd.DataFrame) -> list[list[dict[str, float | str]]]:
    """Compute per-customer top feature impacts using SHAP or safe fallbacks."""
    global _cached_shap_explainer
    if shap is not None:
        try:
            if _cached_shap_explainer is None:
                # Tree models and ensembles benefit from TreeExplainer when available.
                if hasattr(model, "estimators_") or hasattr(model, "get_booster"):
                    _cached_shap_explainer = shap.TreeExplainer(model)
                else:
                    _cached_shap_explainer = shap.Explainer(model, scaled_df)
            shap_values = _cached_shap_explainer(scaled_df).values
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
            logger.info("SHAP impacts computed for %s customers", len(scaled_df))
            return _top_impacts_from_array(shap_values, scaled_df.columns, limit=5)
        except Exception as exc:
            logger.warning("SHAP failed, using fallback impacts. reason=%s", exc)

    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        contribs = scaled_df.to_numpy() * coef
        return _top_impacts_from_array(contribs, scaled_df.columns, limit=5)

    if hasattr(model, "feature_importances_"):
        importances = np.ravel(model.feature_importances_)
        centered = scaled_df.to_numpy() - scaled_df.to_numpy().mean(axis=0)
        contribs = centered * importances
        return _top_impacts_from_array(contribs, scaled_df.columns, limit=5)

    # Final fallback for unsupported ensembles: local effect approximation per feature.
    contribs = np.zeros_like(scaled_df.to_numpy(), dtype=float)
    base_probs = model.predict_proba(scaled_df)[:, 1]
    for col_idx, col_name in enumerate(scaled_df.columns):
        perturbed = scaled_df.copy()
        perturbed[col_name] = 0.0
        perturbed_probs = model.predict_proba(perturbed)[:, 1]
        contribs[:, col_idx] = base_probs - perturbed_probs
    return _top_impacts_from_array(contribs, scaled_df.columns, limit=5)


def _top_impacts_from_array(
    values: np.ndarray, columns: pd.Index, limit: int = 8
) -> list[list[dict[str, float | str]]]:
    """Convert contribution matrix into sorted top-k explainability payload."""
    all_rows: list[list[dict[str, float | str]]] = []
    names = list(columns)
    for row in values:
        indexed = list(enumerate(row))
        indexed.sort(key=lambda item: abs(float(item[1])), reverse=True)
        top = indexed[:limit]
        all_rows.append(
            [
                {
                    "feature": _friendly_feature_name(names[idx]),
                    "impact": round(float(val), 4),
                    "direction": "positive" if val >= 0 else "negative",
                }
                for idx, val in top
            ]
        )
    return all_rows


def _customer_commentary(customer: pd.Series, probability: float, label: str) -> str:
    """Generate dynamic, customer-specific retention commentary text."""
    tenure = float(pd.to_numeric(customer.get("tenure", 0), errors="coerce") or 0)
    monthly = float(pd.to_numeric(customer.get("MonthlyCharges", 0), errors="coerce") or 0)
    internet = str(customer.get("InternetService", "Unknown"))
    contract = str(customer.get("Contract", "Unknown"))
    tech_support = str(customer.get("TechSupport", "Unknown"))

    loyalty = (
        "Bu musteri 24+ aydir bizimle, sadakat seviyesi yuksek."
        if tenure >= 24
        else "Musteri iliskisi erken asamada, baglilik henuz oturmamis."
    )
    risk_driver = (
        f"{internet} internet kullaniminda churn baskisi goruluyor."
        if internet.lower().startswith("fiber")
        else "Baglanti tipine gore risk dengeli gozukuyor."
    )
    support_hint = (
        "Teknik destek paketi olmadigi icin proaktif destek cagrisi onerilir."
        if "no" in tech_support.lower()
        else "Teknik destek mevcut, memnuniyet takibiyle korunabilir."
    )
    contract_hint = (
        "Aylik kontrat churn esnekligini artiriyor; uzun donem teklif uygun."
        if "month-to-month" in contract.lower()
        else "Uzun kontrat etkisi churn riskini dogal olarak baskiliyor."
    )

    return (
        f"{label} segmentinde (%{probability * 100:.1f}) yer aliyor. "
        f"{loyalty} {risk_driver} {contract_hint} {support_hint} "
        f"Aylik odeme seviyesi: {monthly:.2f}."
    )


def _analyze_customers(customers: list[dict[str, Any]]) -> dict[str, Any]:
    """Centralized analyze flow used by JSON and file upload endpoints."""
    try:
        raw_df, scaled_df = _preprocess(customers)
        probabilities = model.predict_proba(scaled_df)[:, 1]
        impacts = _compute_impacts(scaled_df)
    except Exception as exc:
        logger.exception("Analyze failed")
        raise HTTPException(status_code=500, detail=f"Analyze error: {exc}") from exc

    rows = []
    for idx, prob in enumerate(probabilities):
        label = _risk_label(float(prob))
        raw_customer = raw_df.iloc[idx]
        monthly = float(pd.to_numeric(raw_customer.get("MonthlyCharges", 0), errors="coerce") or 0)
        tenure = float(pd.to_numeric(raw_customer.get("tenure", 0), errors="coerce") or 0)
        rows.append(
            {
                "customer_id": str(raw_df.iloc[idx].get("customerID", f"CUST-{idx + 1}")),
                "churn_probability": round(float(prob), 4),
                "will_churn": bool(float(prob) >= THRESHOLD),
                "risk_label": label,
                "threshold": THRESHOLD,
                "top_impacts": impacts[idx],
                "commentary": _customer_commentary(raw_customer, float(prob), label),
                "monthly_charges": round(monthly, 2),
                "tenure": round(tenure, 2),
            }
        )

    logger.info("Analyze success. customers=%s", len(rows))
    return {"threshold": THRESHOLD, "count": len(rows), "results": rows}


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status for monitoring checks."""
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    """Analyze customers and return churn probability plus AI insights."""
    return _analyze_customers(payload.customers)


@app.post("/api/analyze-file")
async def analyze_file(file: UploadFile = File(...)) -> dict[str, Any]:
    """Analyze customers from uploaded CSV, JSON, or Excel file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")
    content = await file.read()
    customers = _parse_uploaded_file(file.filename, content)
    logger.info("Analyze file upload. file=%s rows=%s", file.filename, len(customers))
    return _analyze_customers(customers)


@app.post("/api/coupon")
def generate_coupon(payload: CouponRequest) -> dict[str, str | int]:
    """Generate ARPU-aware retention coupon recommendations."""
    risk = payload.risk_label.lower()
    if payload.monthly_charges >= 85:
        discount = 30
    elif payload.monthly_charges >= 55:
        discount = 20
    else:
        discount = 10

    if "kritik" in risk and discount < 20:
        discount = 30

    code = f"RET-{payload.customer_id[-4:].upper()}-{secrets.token_hex(2).upper()}"
    message = (
        f"{discount}% teklif uretildi. "
        f"ARPU={payload.monthly_charges:.2f}, tenure={payload.tenure:.1f} ay, segment={payload.risk_label}."
    )
    logger.info("Coupon generated. customer=%s discount=%s", payload.customer_id, discount)
    return {
        "customer_id": payload.customer_id,
        "coupon_code": code,
        "discount_percent": discount,
        "message": message,
    }


@app.exception_handler(Exception)
def unhandled_exception_handler(_, exc: Exception) -> JSONResponse:
    """Return consistent JSON errors for uncaught server exceptions."""
    logger.exception("Unhandled server error")
    return JSONResponse(status_code=500, content={"detail": f"Unexpected error: {exc}"})
