import sys
import uuid
from pathlib import Path

# streamlit run adds this script's own directory to sys.path, not the repo
# root - add the root explicitly so `config`/`core`/`db`/`prompts` imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app.components.chat import render_history, render_message  # noqa: E402
from app.components.feedback_widget import render_feedback_widget  # noqa: E402
from app.components.sidebar import render_sidebar  # noqa: E402
from core.chains.interpretation import build_interpretation_chain  # noqa: E402
from core.chains.pipeline import answer_question  # noqa: E402
from core.chains.retrieval import build_retrieval_chain  # noqa: E402
from core.context.history import build_history  # noqa: E402
from db.engine import init_db  # noqa: E402
from db.repositories import query_repo, session_repo  # noqa: E402

st.title("MItra Query Hub 🔍")

init_db()


def resolve_session_id() -> str:
    session_id = st.query_params.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        st.query_params["session_id"] = session_id
    session_repo.get_or_create(session_id)
    return session_id


@st.cache_resource
def get_chains():
    return build_retrieval_chain(), build_interpretation_chain()


def load_display_messages(session_id: str) -> list:
    display = []
    for query in query_repo.get_recent_for_session(session_id):
        display.append({"role": "user", "content": query.user_query, "query_id": None})
        display.append({"role": "assistant", "content": query.ai_response, "query_id": query.query_id})
    return display


session_id = resolve_session_id()
st.session_state["session_id"] = session_id
render_sidebar()

retrieval_chain, interpretation_chain = get_chains()

messages = load_display_messages(session_id)
render_history(messages)
for message in messages:
    if message["role"] == "assistant" and message["query_id"]:
        render_feedback_widget(message["query_id"])

if prompt := st.chat_input("Ask a question about the MItra data..."):
    render_message("user", prompt)

    with st.spinner("Generating response..."):
        history = build_history(query_repo.get_recent_for_session(session_id))
        result = answer_question(retrieval_chain, interpretation_chain, prompt, history.messages)
        query_id = query_repo.create_query(
            session_id=session_id,
            user_query=prompt,
            generated_sql=result["generated_sql"],
            ai_response=result["answer"],
        )

    render_message("assistant", result["answer"])
    render_feedback_widget(query_id)
