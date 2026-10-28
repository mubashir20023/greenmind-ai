import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Connection string comes from the environment (see .env.example).
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:@localhost/plant-fyp")

# Create engine (set SQL_ECHO=1 to log SQL statements)
engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=os.getenv("SQL_ECHO", "0") == "1", pool_pre_ping=True)

# Session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()
