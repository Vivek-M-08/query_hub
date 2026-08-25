import streamlit as st

from db.repositories.feedback_repo import create_feedback


def render_feedback_widget(query_id: str) -> None:
    """Thumbs up/down + optional comment for one assistant response, keyed by query_id."""
    submitted_key = f"feedback_submitted_{query_id}"
    if st.session_state.get(submitted_key):
        st.caption("Thanks for the feedback!")
        return

    cols = st.columns([1, 1, 10])

    if cols[0].button("👍", key=f"up_{query_id}"):
        create_feedback(query_id, rating="up")
        st.session_state[submitted_key] = True
        st.rerun()

    if cols[1].button("👎", key=f"down_{query_id}"):
        st.session_state[f"show_comment_{query_id}"] = True

    if st.session_state.get(f"show_comment_{query_id}"):
        comment = st.text_input("What could be improved? (optional)", key=f"comment_{query_id}")
        if st.button("Submit", key=f"submit_{query_id}"):
            create_feedback(query_id, rating="down", comment=comment or None)
            st.session_state[submitted_key] = True
            st.rerun()
