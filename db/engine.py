from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config.settings import get_app_db_url

engine = create_engine(get_app_db_url())
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def init_db() -> None:
    """Create all app-DB tables if they don't already exist."""
    from db import models  # noqa: F401 - registers models on Base before create_all

    Base.metadata.create_all(engine)
