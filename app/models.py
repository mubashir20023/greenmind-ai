from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
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

    # ✅ THESE WERE MISSING!
    images = relationship("PlantImage", back_populates="plant")
    advices = relationship("PlantAdvice", back_populates="plant")


class PlantImage(Base):
    __tablename__ = "plant_images"

    id = Column(Integer, primary_key=True, index=True)
    plant_id = Column(Integer, ForeignKey("plants.id"))
    image_path = Column(String(255))
    timestamp = Column(DateTime, default=datetime.utcnow)

    plant = relationship("Plant", back_populates="images")


class PlantAdvice(Base):
    __tablename__ = "plant_advices"

    id = Column(Integer, primary_key=True, index=True)
    plant_id = Column(Integer, ForeignKey("plants.id"))
    advice_text = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

    plant = relationship("Plant", back_populates="advices")


# class User(Base):
#     __tablename__ = "users"

#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String(100))
#     email = Column(String(255), unique=True, index=True)
#     password_hash = Column(String(255))

#     plants = relationship("Plant", back_populates="user")

class User(Base, UserMixin):  # ✅ Inherit from UserMixin
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    email = Column(String(255), unique=True, index=True)
    password_hash = Column(String(255))

    plants = relationship("Plant", back_populates="user")