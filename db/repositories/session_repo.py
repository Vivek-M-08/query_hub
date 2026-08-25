from db.engine import SessionLocal
from db.models import Session as SessionModel


def get_or_create(session_id: str) -> str:
    """Ensure a sessions row exists for this session_id, returning it either way."""
    with SessionLocal() as db:
        existing = db.get(SessionModel, session_id)
        if existing:
            return existing.session_id

        new_session = SessionModel(session_id=session_id)
        db.add(new_session)
        db.commit()
        return new_session.session_id
