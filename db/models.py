import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from db.engine import Base


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Session(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Query(Base):
    __tablename__ = "queries"

    query_id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.session_id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    user_query: Mapped[str] = mapped_column(Text)
    generated_sql: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_response: Mapped[str] = mapped_column(Text)


class Feedback(Base):
    __tablename__ = "feedback"

    feedback_id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    query_id: Mapped[str] = mapped_column(String, ForeignKey("queries.query_id"))
    rating: Mapped[str] = mapped_column(String)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
