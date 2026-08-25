import html

import streamlit as st

USER_BUBBLE_COLOR = "#DCF8C6"
BOT_BUBBLE_COLOR = "#F1F0F0"


def render_message(role: str, content: str) -> None:
    """Render a chat message as a right-aligned (user) or left-aligned (bot) bubble.

    Rendered as plain flexbox HTML rather than st.chat_message: Streamlit's
    chat_message has no built-in side-alignment, and its internal DOM/testids
    aren't a stable API to build CSS selectors against (requirements.txt pins
    no streamlit version). This has no such dependency.
    """
    is_user = role == "user"
    justify = "flex-end" if is_user else "flex-start"
    background = USER_BUBBLE_COLOR if is_user else BOT_BUBBLE_COLOR

    st.markdown(
        f"""
        <div style="display:flex; justify-content:{justify}; margin:6px 0;">
          <div style="max-width:70%; padding:10px 14px; border-radius:14px;
                      background:{background}; white-space:pre-wrap; word-wrap:break-word;">
            {html.escape(content)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history(messages) -> None:
    for message in messages:
        render_message(message["role"], message["content"])
