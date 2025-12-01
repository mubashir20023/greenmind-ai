from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime
from flask_login import UserMixin


class Plant(Base):
    __tablename__ = "plants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    description = Column(String(500))

    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="plants")
    # NEW FIELD (MAIN FIX)
    species = Column(String(255), index=True)   # <-- Add this
    common_group = Column(String(255), index=True)  # optional but recommended
    
    # relationships
    images = relationship("PlantPhoto", back_populates="plant", cascade="all, delete-orphan")
    advices = relationship("PlantAdvice", back_populates="plant", cascade="all, delete-orphan")
    healths = relationship("PlantHealth", back_populates="plant", cascade="all, delete-orphan")


class PlantPhoto(Base):
    __tablename__ = "plant_photos"

    id = Column(Integer, primary_key=True)
    plant_id = Column(Integer, ForeignKey("plants.id"), nullable=False, index=True)
    photo_url = Column(String(255), nullable=False)
    uploaded_on = Column(DateTime, default=datetime.utcnow, nullable=False)
    health_status = Column(String(50), nullable=True)

    plant = relationship("Plant", back_populates="images")


class PlantAdvice(Base):
    __tablename__ = "plant_advices"

    id = Column(Integer, primary_key=True, index=True)
    plant_id = Column(Integer, ForeignKey("plants.id"))
    advice_text = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

    plant = relationship("Plant", back_populates="advices")


class User(Base, UserMixin):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    email = Column(String(255), unique=True, index=True)
    password_hash = Column(String(255))

    plants = relationship("Plant", back_populates="user")


class PlantHealth(Base):
    __tablename__ = "plant_health"

    id = Column(Integer, primary_key=True, index=True)
    plant_id = Column(Integer, ForeignKey("plants.id"))
    payload = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    plant = relationship("Plant", back_populates="healths")
