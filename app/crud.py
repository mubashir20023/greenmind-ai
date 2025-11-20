from sqlalchemy.orm import Session
from app import models

def create_plant(db: Session, name: str, species: str):
    plant = models.Plant(name=name, species=species)
    db.add(plant)
    db.commit()
    db.refresh(plant)
    return plant

def get_plants(db: Session, skip: int = 0, limit: int = 10):
    return db.query(models.Plant).offset(skip).limit(limit).all()
