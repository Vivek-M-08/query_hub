import os

import streamlit as st


def render_sidebar() -> None:
    st.sidebar.markdown("### Connected to")
    st.sidebar.text(f"Database: {os.getenv('ANALYTICS_DB_NAME', 'analytics_db')}")
    st.sidebar.text(f"Model: {os.getenv('INTERPRETATION_MODEL', 'anthropic/claude-sonnet-5')}")
    st.sidebar.divider()
    session_id = st.session_state.get("session_id", "")
    st.sidebar.caption(f"Session: {session_id[:8]}…")
