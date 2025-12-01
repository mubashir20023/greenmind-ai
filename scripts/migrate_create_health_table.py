# scripts/migrate_create_health_table.py
from app.database import engine, Base
import app.models  # ensures all models are loaded

if __name__ == "__main__":
    print("Creating tables (if not exist)...")
    Base.metadata.create_all(bind=engine)
    print("Done.")
