import json
import os
import sys
from functools import lru_cache
from typing import List

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
ML_TRAINING_DIR = os.getenv("ML_TRAINING_DIR", os.path.join(REPO_ROOT, "ml_training"))

if ML_TRAINING_DIR not in sys.path:
    sys.path.insert(0, ML_TRAINING_DIR)

from app.core.database import Base, engine, get_db
from app.models import Screening, User
from app.schemas import (
    AuthResponse,
    LoginRequest,
    PatientListItem,
    PredictRequestSchema,
    RegisterRequest,
    ScreeningResponse,
    UserResponse,
)
from app.security import create_access_token, decode_access_token, hash_password, verify_password
from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


@lru_cache(maxsize=1)
def get_predictor() -> PredictPipeline:
    predictor = PredictPipeline()
    artifacts_dir = os.path.join(ML_TRAINING_DIR, "artifacts")
    predictor.config.model_path = os.path.join(artifacts_dir, "model.pkl")
    predictor.config.preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
    predictor.config.metrics_path = os.path.join(artifacts_dir, "metrics.json")
    predictor.config.feature_names_path = os.path.join(artifacts_dir, "feature_names.json")
    predictor.config.shap_summary_path = os.path.join(artifacts_dir, "shap_summary.json")
    return predictor


@app.get("/")
def root():
    return {"message": "backend is running"}


@app.get("/explainability/summary")
def explainability_summary():
    predictor = get_predictor()
    summary_path = predictor.config.shap_summary_path
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="SHAP summary not found. Run training pipeline first.")

    with open(summary_path, "r", encoding="utf-8") as shap_file:
        shap_summary = json.load(shap_file)

    top_features = []
    for item in shap_summary.get("top_features_by_mean_abs_shap", [])[:8]:
        feature = str(item.get("feature"))
        label = predictor._friendly_feature_label(feature)
        top_features.append(
            {
                "feature": feature,
                "feature_label": label,
                "impact_score": item.get("mean_abs_shap"),
                "plain_reason": predictor._driver_reason(label),
            }
        )

    return {
        "status": shap_summary.get("status", "unknown"),
        "sample_size": shap_summary.get("sample_size"),
        "top_features": top_features,
    }


def _build_auth_response(user: User) -> AuthResponse:
    token = create_access_token(subject=user.email, user_id=user.id, role=user.role)
    return AuthResponse(
        access_token=token,
        user=UserResponse(id=user.id, full_name=user.full_name, email=user.email, role=user.role),
    )


def _register_user(payload: RegisterRequest, role: str, db: Session) -> AuthResponse:
    existing_user = db.query(User).filter(User.email == payload.email.lower()).first()
    if existing_user:
        raise HTTPException(status_code=409, detail="User already exists with this email.")

    user = User(
        full_name=payload.full_name.strip(),
        email=payload.email.lower().strip(),
        role=role,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _build_auth_response(user)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        payload = decode_access_token(token)
        user_id = int(payload.get("uid", 0))
        email = str(payload.get("sub", "")).lower()
        role = str(payload.get("role", "")).lower()
    except (ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    user = db.query(User).filter(User.id == user_id, User.email == email, User.role == role).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials.")
    return user


def require_roles(*allowed_roles: str):
    def _role_dependency(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(status_code=403, detail="You are not allowed to access this resource.")
        return current_user

    return _role_dependency


def _screening_to_response(screening: Screening) -> ScreeningResponse:
    return ScreeningResponse(
        id=screening.id,
        patient_id=screening.patient_id,
        risk_probability=screening.risk_probability,
        risk_category=screening.risk_category,
        predicted_class=screening.predicted_class,
        decision_threshold=screening.decision_threshold,
        input_payload=screening.input_payload,
        prediction_payload=screening.prediction_payload,
        created_at=screening.created_at,
    )


@app.post("/auth/register", response_model=AuthResponse)
def register_default_patient(payload: RegisterRequest, db: Session = Depends(get_db)):
    return _register_user(payload, role="patient", db=db)


@app.post("/auth/patient/register", response_model=AuthResponse)
def register_patient(payload: RegisterRequest, db: Session = Depends(get_db)):
    return _register_user(payload, role="patient", db=db)


@app.post("/auth/doctor/register", response_model=AuthResponse)
def register_doctor(payload: RegisterRequest, db: Session = Depends(get_db)):
    return _register_user(payload, role="doctor", db=db)


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return _build_auth_response(user)


@app.get("/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse(
        id=current_user.id,
        full_name=current_user.full_name,
        email=current_user.email,
        role=current_user.role,
    )


@app.post("/predict")
def predict(payload: PredictRequestSchema, current_user: User = Depends(require_roles("patient", "doctor"))):
    try:
        input_df = pd.DataFrame([payload.model_dump()])
        prediction = get_predictor().predict(input_df)
        return prediction
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/screenings", response_model=ScreeningResponse)
def create_screening(
    payload: PredictRequestSchema,
    current_user: User = Depends(require_roles("patient")),
    db: Session = Depends(get_db),
):
    try:
        input_payload = payload.model_dump()
        prediction_payload = get_predictor().predict(pd.DataFrame([input_payload]))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    screening = Screening(
        patient_id=current_user.id,
        risk_probability=float(prediction_payload["risk_probability"]),
        risk_category=str(prediction_payload["risk_category"]),
        predicted_class=int(prediction_payload["predicted_class"]),
        decision_threshold=float(prediction_payload["decision_threshold"]),
        input_payload=input_payload,
        prediction_payload=prediction_payload,
    )
    db.add(screening)
    db.commit()
    db.refresh(screening)
    return _screening_to_response(screening)


@app.get("/screenings/me", response_model=List[ScreeningResponse])
def my_screenings(
    current_user: User = Depends(require_roles("patient")),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Screening)
        .filter(Screening.patient_id == current_user.id)
        .order_by(Screening.created_at.desc())
        .all()
    )
    return [_screening_to_response(row) for row in rows]


@app.get("/doctor/patients", response_model=List[PatientListItem])
def doctor_list_patients(
    current_user: User = Depends(require_roles("doctor")),
    db: Session = Depends(get_db),
):
    _ = current_user
    rows = db.query(User).filter(User.role == "patient").order_by(User.created_at.desc()).all()
    return [
        PatientListItem(
            id=row.id,
            full_name=row.full_name,
            email=row.email,
            role=row.role,  # type: ignore[arg-type]
            created_at=row.created_at,
        )
        for row in rows
    ]


@app.get("/doctor/patients/{patient_id}/screenings", response_model=List[ScreeningResponse])
def doctor_patient_screenings(
    patient_id: int,
    current_user: User = Depends(require_roles("doctor")),
    db: Session = Depends(get_db),
):
    _ = current_user
    patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    rows = db.query(Screening).filter(Screening.patient_id == patient_id).order_by(Screening.created_at.desc()).all()
    return [_screening_to_response(row) for row in rows]
