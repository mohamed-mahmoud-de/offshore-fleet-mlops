"""
Database connection helper — the single source of truth for how we reach Postgres.

Every other module imports get_engine() from here, so the connection details live
in exactly one place and the password is only ever read from .env (never hardcoded).
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Load .env from the project root, no matter which folder we're run from.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")


def get_engine() -> Engine:
    """Build a SQLAlchemy engine from the POSTGRES_* variables in .env."""
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host = os.environ["POSTGRES_HOST"]
    port = os.environ["POSTGRES_PORT"]
    db = os.environ["POSTGRES_DB"]

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)
