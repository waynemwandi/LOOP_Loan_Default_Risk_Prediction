import csv
import io
import logging
import os
import pickle
import traceback
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .loop_auth import LoopAuth
from .loop_payout import LoopPayout

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

today = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.path.join(LOG_DIR, f"{today}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),  # console too
    ],
)

logger = logging.getLogger("loop-risk")

app = FastAPI(title="LOOP Risk API")
loop_auth = LoopAuth()
loop_pay = LoopPayout(loop_auth)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount(
    "/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static"
)

# Load .env vars
MODEL_PATH = os.getenv(
    "MODEL_PATH", os.path.join(BASE_DIR, "models", "loan_default_xgboost_model.pkl")
)
MODEL_VERSION = os.getenv("MODEL_VERSION", "N/A")

MODEL: Any | None = None
MODEL_LOAD_ERROR: str | None = None


try:
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)
        logger.info("Loaded model from %s", MODEL_PATH)
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    logger.error("Failed to load model from %s: %s", MODEL_PATH, MODEL_LOAD_ERROR)

# Helper functions


def expected_feature_names() -> list[str] | None:
    """Best-effort list of feature names expected by the model."""
    # Make Pylance happy and avoid attribute access on None
    m = MODEL
    if m is None:
        return None

    try:
        # 1) Plain sklearn estimator trained with feature_names_in_
        if hasattr(m, "feature_names_in_"):
            return list(getattr(m, "feature_names_in_"))

        # 2) sklearn Pipeline: search each step for feature_names_in_
        if hasattr(m, "named_steps") and getattr(m, "named_steps", None):
            for step in m.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    return list(getattr(step, "feature_names_in_"))

        # 3) xgboost models: read from the booster
        if hasattr(m, "get_booster"):
            booster = m.get_booster()
            names = getattr(booster, "feature_names", None)
            if names:
                return list(names)
    except Exception as exc:
        # Keep running even if introspection fails
        logger.debug("expected_feature_names() failed: %s", exc)

    return None


def build_feature_row(
    age,
    credit_score,
    no_default_loan,
    net_income,
    principal_disbursed,
    emi,
    gender,
    marital_status,
    product,
) -> pd.DataFrame:
    # Normalize categoricals to uppercase (common in training)
    gender = (gender or "").upper()
    marital_status = (marital_status or "").upper()
    product = (product or "").upper()

    # Provide BOTH snake_case and TRAINING-style keys (if they used originals)
    row = {
        # snake_case
        "age": age,
        "credit_score": credit_score,
        "no_default_loan": no_default_loan,
        "net_income": net_income,
        "principal_disbursed": principal_disbursed,
        "emi": emi,
        "gender": gender,
        "marital_status": marital_status,
        "product": product,
        # common training keys seen in your teammate’s form
        "AGE": age,
        "CREDIT_SCORE": credit_score,
        "NO_DEFAULT_LOAN": no_default_loan,
        "NET INCOME": net_income,
        "PRINCIPAL_DISBURSED": principal_disbursed,
        "EMI": emi,
        "GENDER": gender,
        "MARITAL_STATUS": marital_status,
        "PRODUCT": product,
    }
    return pd.DataFrame([row])


def score_proba(df: pd.DataFrame) -> float:
    """Return a probability in [0,1] using the loaded model (predict_proba preferred)."""
    if MODEL is None:
        raise RuntimeError(f"Model not loaded: {MODEL_LOAD_ERROR}")

    logger.info("Scoring with model type: %s", type(MODEL))

    if hasattr(MODEL, "predict_proba"):
        logger.info("Using predict_proba")
        proba = MODEL.predict_proba(df)
        logger.info(
            "predict_proba output shape=%s value=%s",
            getattr(proba, "shape", None),
            proba,
        )
        return float(np.array(proba).reshape(1, -1)[0, -1])

    if hasattr(MODEL, "predict"):
        logger.info("Using predict")
        y = np.array(MODEL.predict(df)).reshape(-1)
        logger.info("predict output=%s", y)
        val = float(y[0])
        if 0.0 <= val <= 1.0:
            return val
        # assume log-odds
        return 1.0 / (1.0 + np.exp(-val))
    raise RuntimeError("Model has neither predict_proba nor predict.")


# Endpoints


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "model_version": MODEL_VERSION}
    )


@app.post("/score")
async def score(
    request: Request,
    # existing-customer path
    customer_id: str | None = Form(None),
    # new-applicant fields
    age: int | None = Form(None),
    credit_score: float | None = Form(None),
    no_default_loan: int | None = Form(None),
    net_income: float | None = Form(None),
    principal_disbursed: float | None = Form(None),
    emi: float | None = Form(None),
    gender: str | None = Form(None),
    marital_status: str | None = Form(None),
    product: str | None = Form(None),
    phone: str | None = Form(None),
):
    # --- Existing-customer path (stub) ---
    if customer_id:
        dummy_result = {
            "probability": 0.27,
            "band": "Low",
            "reasons": ["Demo (existing-customer path)"],
            "inputs": {"customer_id": customer_id},
        }
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "result": dummy_result,
                "model_version": MODEL_VERSION,
            },
        )

    # --- New-applicant path (run through the model) ---
    reasons = []
    try:
        # 1) Build a one-row DataFrame from form inputs
        df = build_feature_row(
            age,
            credit_score,
            no_default_loan,
            net_income,
            principal_disbursed,
            emi,
            gender,
            marital_status,
            product,
        )
        logger.info(
            "Manual submit payload (raw): %s",
            {
                "age": age,
                "credit_score": credit_score,
                "no_default_loan": no_default_loan,
                "net_income": net_income,
                "principal_disbursed": principal_disbursed,
                "emi": emi,
                "gender": gender,
                "marital_status": marital_status,
                "product": product,
            },
        )

        # 2) If your model DOES NOT include preprocessing and expects numerics only,
        #    uncomment the encoders below. If it DOES include preprocessing, keep them commented.
        # --- BEGIN OPTIONAL ENCODERS ---
        # gender_map = {"MALE": 1, "FEMALE": 0}
        # marital_map = {"SINGLE": 0, "MARRIED": 1, "OTHER": 2}
        # product_map = {
        #     "PERSONAL UNSECURED SCHEME LOAN": 0,
        #     "INDIVIDUAL IPF": 1,
        #     "MOBILE LOAN": 2,
        #     "COMMERCIAL VEHICLES": 3,
        #     "DIGITAL PERSONAL LOAN": 4,
        # }
        # if "GENDER" in df.columns: df["GENDER"] = df["GENDER"].map(gender_map).astype("float64")
        # if "MARITAL_STATUS" in df.columns: df["MARITAL_STATUS"] = df["MARITAL_STATUS"].map(marital_map).astype("float64")
        # if "PRODUCT" in df.columns: df["PRODUCT"] = df["PRODUCT"].map(product_map).astype("float64")
        # --- END OPTIONAL ENCODERS ---

        # 3) Force DataFrame to the exact columns/order the model expects (if discoverable)
        exp = expected_feature_names()
        if exp:
            for c in exp:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[exp]
            logger.info("Aligned DF to expected features (%d): %s", len(exp), exp)
        else:
            # Fallback to teammate’s training-style names and order
            training_cols = [
                "AGE",
                "CREDIT_SCORE",
                "NO_DEFAULT_LOAN",
                "NET INCOME",
                "PRINCIPAL_DISBURSED",
                "EMI",
                "GENDER",
                "MARITAL_STATUS",
                "PRODUCT",
            ]
            for c in training_cols:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[training_cols]
            logger.info("Aligned DF to fallback training columns: %s", training_cols)

        logger.info(
            "DataFrame to model -> cols: %s | dtypes: %s | head: %s",
            list(df.columns),
            df.dtypes.astype(str).to_dict(),
            df.head(1).to_dict(orient="records"),
        )

        # 4) Score
        pd_val = score_proba(df)
        prob = max(0.0, min(1.0, float(pd_val)))  # clamp 0–1
        band = "Low" if prob < 0.30 else ("Medium" if prob < 0.70 else "High")
        reasons.append(f"Scored via {os.path.basename(MODEL_PATH)} (v{MODEL_VERSION})")

        # --- Auto-approve and pay for Low risk ---
        transfer = None
        if (
            band == "Low"
            and phone
            and principal_disbursed
            and float(principal_disbursed) > 0
        ):
            try:
                loop_pay.get_user_detail(phone)  # optional

                resp = loop_pay.funds_transfer(
                    amount=float(principal_disbursed),
                    recipient_mobile=phone,
                    recipient_account_type="MOMO",
                    recipient_bank_code=None,  # env default
                    recipient_name=None,
                    narration=f"Loan disbursement to {phone}",
                )

                # Normalize for the template (result.transfer.*)
                def _ok(r: dict) -> bool:
                    code = (str(r.get("responseCode") or "")).strip()
                    status = (r.get("status") or "").upper().strip()
                    return status in {"SUCCESS", "OK"} or code in {"00", "000"}

                transfer = {
                    "ok": _ok(resp),
                    "status": resp.get("status")
                    or resp.get("responseDescription")
                    or resp.get("message"),
                    "amount": float(principal_disbursed),
                    # Your template checks these keys in this order; make sure one is present
                    "transferRefNo": resp.get("transferRefNo")
                    or resp.get("transactionReferenceNo")
                    or resp.get("requestId"),
                    "transactionReferenceNo": resp.get("transactionReferenceNo"),
                    "requestId": resp.get("requestId"),
                    "transferType": resp.get("transferType") or "20",
                    "transferChannel": resp.get("transferChannel") or "65",
                    "dry_run": bool(os.getenv("DRY_RUN", "").lower() == "true"),
                }

                logger.info("Normalized transfer for template: %s", transfer)
                reasons.append("Auto-approved and queued payout (PesaLink-mobile)")

            except Exception as pay_err:
                logger.exception("Payout failed")
                reasons.append(f"Payout error: {type(pay_err).__name__}")

    except Exception as e:
        # Graceful fallback with detailed logging
        tb = traceback.format_exc()
        logger.error("Model scoring failed: %s\n%s", e, tb)
        prob = 0.27
        band = "Low"
        msg = str(e)
        if len(msg) > 180:
            msg = msg[:180] + "..."
        reasons += [
            f"Model error: {e.__class__.__name__}",
            msg,
            "Falling back to demo score",
        ]

    # 5) Render
    result = {
        "probability": prob,
        "band": band,
        "reasons": reasons,
        "inputs": {
            "age": age,
            "credit_score": credit_score,
            "no_default_loan": no_default_loan,
            "net_income": net_income,
            "principal_disbursed": principal_disbursed,
            "emi": emi,
            "gender": gender,
            "marital_status": marital_status,
            "product": product,
            "phone": phone,
        },
        "transfer": transfer,
    }

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "result": result, "model_version": MODEL_VERSION},
    )


# add this new handler under the existing routes
@app.post("/score-csv")
async def score_csv(request: Request, file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if not name.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw = await file.read()
    text = raw.decode("utf-8-sig", errors="ignore")  # handle BOM safely

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV has no data rows")

    # For now: dummy scoring for each row
    results = []
    for r in rows:
        results.append(
            {
                "probability": 0.27,  # static demo PD
                "band": "Low",
                "reasons": ["Demo scoring only"],
                "inputs": r,
            }
        )

    summary = {
        "filename": file.filename,
        "rows": len(rows),
        "columns": list(reader.fieldnames or []),
    }

    # Preview only first 10 rows to keep the page small
    preview = results[:10]

    return templates.TemplateResponse(
        "batch_result.html",
        {
            "request": request,
            "summary": summary,
            "preview": preview,
            "model_version": MODEL_VERSION,
        },
    )


@app.get("/debug-model")
async def debug_model():
    return {
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH,
        "model_version": MODEL_VERSION,
        "load_error": MODEL_LOAD_ERROR,
        "type": str(type(MODEL)) if MODEL is not None else None,
        "expected_features": expected_feature_names(),
    }


@app.get("/loop/token")
async def loop_token_probe():
    try:
        header = loop_auth.get_token()
        # redact most of it in logs and response
        redacted = header.split(" ", 1)[1]
        shown = redacted[:10] + "..." if len(redacted) > 10 else "..."
        return {"ok": True, "token_preview": shown}
    except Exception as e:
        logger.exception("LOOP token probe failed")
        return {"ok": False, "error": str(e)}


@app.get("/health")
async def health():
    return {
        "status": "ok" if MODEL is not None else "degraded",
        "model_version": MODEL_VERSION,
        "model_path": MODEL_PATH,
        "load_error": MODEL_LOAD_ERROR,
    }
