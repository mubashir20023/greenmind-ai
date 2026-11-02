import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Connection string comes from the environment (see .env.example).
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:@localhost/plant-fyp")

# Create engine (set SQL_ECHO=1 to log SQL statements).
# SQLite (e.g. DATABASE_URL=sqlite:///data/greenmind.db, used on Hugging Face Spaces) needs
# check_same_thread=False because Flask serves requests from several threads.
_connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=os.getenv("SQL_ECHO", "0") == "1",
                       pool_pre_ping=True, connect_args=_connect_args)

# Session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()
