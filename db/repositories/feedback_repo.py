from typing import Optional

from db.engine import SessionLocal
from db.models import Feedback


def create_feedback(query_id: str, rating: str, comment: Optional[str] = None) -> str:
    with SessionLocal() as db:
        feedback = Feedback(query_id=query_id, rating=rating, comment=comment)
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        return feedback.feedback_id
