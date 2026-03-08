from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship

from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    role = Column(String(20), index=True, nullable=False)  # patient | doctor
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    screenings = relationship("Screening", back_populates="patient", cascade="all, delete-orphan")


class Screening(Base):
    __tablename__ = "screenings"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    risk_probability = Column(Float, nullable=False)
    risk_category = Column(String(32), nullable=False)
    predicted_class = Column(Integer, nullable=False)
    decision_threshold = Column(Float, nullable=False)
    input_payload = Column(JSON, nullable=False)
    prediction_payload = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    patient = relationship("User", back_populates="screenings")
