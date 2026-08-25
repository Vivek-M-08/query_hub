import os
import urllib.parse

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RETRIEVAL_MODEL = os.getenv("RETRIEVAL_MODEL", "anthropic/claude-sonnet-5")
INTERPRETATION_MODEL = os.getenv("INTERPRETATION_MODEL", "anthropic/claude-sonnet-5")

LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")


def _build_postgres_url(prefix: str, default_name: str) -> str:
    """Build a SQLAlchemy Postgres URL from a fully independent set of
    {prefix}_DB_USER/PASSWORD/HOST/PORT/NAME env vars - no assumption that
    different databases share a server or credentials.
    """
    db_user = os.getenv(f"{prefix}_DB_USER")
    db_host = os.getenv(f"{prefix}_DB_HOST")
    db_port = os.getenv(f"{prefix}_DB_PORT")
    db_name = os.getenv(f"{prefix}_DB_NAME", default_name)
    db_password = urllib.parse.quote(os.getenv(f"{prefix}_DB_PASSWORD", ""))

    if not all([db_user, db_host, db_port, db_name]):
        raise ValueError(f"{prefix} DB configuration missing from environment variables.")

    return f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def get_analytics_db_url() -> str:
    """Connection string for analytics_db - the MItra data being queried."""
    return _build_postgres_url("ANALYTICS", default_name="analytics_db")


def get_app_db_url() -> str:
    """Connection string for the app's own database - sessions/queries/feedback."""
    return _build_postgres_url("APP", default_name="query_hub")
