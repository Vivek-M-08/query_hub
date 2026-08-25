from typing import List, Optional

from db.engine import SessionLocal
from db.models import Query


def create_query(
    session_id: str,
    user_query: str,
    generated_sql: Optional[str],
    ai_response: str,
) -> str:
    with SessionLocal() as db:
        query = Query(
            session_id=session_id,
            user_query=user_query,
            generated_sql=generated_sql,
            ai_response=ai_response,
        )
        db.add(query)
        db.commit()
        db.refresh(query)
        return query.query_id


def get_recent_for_session(session_id: str, limit: int = 20) -> List[Query]:
    with SessionLocal() as db:
        return (
            db.query(Query)
            .filter(Query.session_id == session_id)
            .order_by(Query.timestamp.asc())
            .limit(limit)
            .all()
        )
