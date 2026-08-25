from langchain.memory import ChatMessageHistory


def build_history(session_queries) -> ChatMessageHistory:
    """Rebuild LangChain message history from persisted queries for a session."""
    history = ChatMessageHistory()
    for query in session_queries:
        history.add_user_message(query.user_query)
        history.add_ai_message(query.ai_response)
    return history
