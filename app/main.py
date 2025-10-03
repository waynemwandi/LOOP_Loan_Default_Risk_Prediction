import csv
import io
import os

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="LOOP Risk Demo")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount(
    "/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static"
)

MODEL_VERSION = os.getenv("MODEL_VERSION", "N/A")  # placeholder until you load a .pkl


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "model_version": MODEL_VERSION}
    )


from fastapi import Form


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
):
    dummy_result = {
        "probability": 0.27,
        "band": "Low",
        "reasons": ["Demo only"],
        "inputs": {
            "customer_id": customer_id,
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
    }
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "result": dummy_result, "model_version": MODEL_VERSION},
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


@app.get("/health")
async def health():
    return {"status": "ok", "model_version": MODEL_VERSION}
