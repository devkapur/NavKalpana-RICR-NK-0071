from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Role = Literal["patient", "doctor"]


class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=120)
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)


class UserResponse(BaseModel):
    id: int
    full_name: str
    email: str
    role: Role


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class PredictRequestSchema(BaseModel):
    age: int = Field(..., description="Age in days")
    gender: int = Field(..., description="1=female, 2=male as per training data")
    height: float
    weight: float
    ap_hi: float
    ap_lo: float
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int
    id: Optional[int] = None


class ScreeningResponse(BaseModel):
    id: int
    patient_id: int
    risk_probability: float
    risk_category: str
    predicted_class: int
    decision_threshold: float
    input_payload: Dict[str, Any]
    prediction_payload: Dict[str, Any]
    created_at: datetime


class PatientListItem(BaseModel):
    id: int
    full_name: str
    email: str
    role: Role
    created_at: datetime
