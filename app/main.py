import os

from fastapi import FastAPI, Form, Request
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


@app.post("/score")
async def score(
    request: Request,
    # Existing customer path
    customer_id: str | None = Form(None),
    # New applicant path (keep minimal for now)
    age: int | None = Form(None),
    income: float | None = Form(None),
):
    # For now, return a static/dummy result so we can test rendering
    dummy_result = {
        "probability": 0.27,  # pretend PD (27%)
        "band": "Low",  # mapped risk band
        "reasons": ["Stable income", "No recent delinquencies", "Good payment history"],
        "inputs": {
            "customer_id": customer_id,
            "age": age,
            "income": income,
        },
    }
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "result": dummy_result, "model_version": MODEL_VERSION},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_version": MODEL_VERSION}
    return {"status": "ok", "model_version": MODEL_VERSION}
    return {"status": "ok", "model_version": MODEL_VERSION}
