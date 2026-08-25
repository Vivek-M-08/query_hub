from langchain_openai import ChatOpenAI

from config.settings import INTERPRETATION_MODEL, OPENROUTER_API_KEY, RETRIEVAL_MODEL

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _build_llm(model_name: str, temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=model_name,
        temperature=temperature,
        default_headers={"HTTP-Referer": "query-hub-poc", "X-Title": "MItra Query Hub"},
    )


def get_retrieval_llm() -> ChatOpenAI:
    return _build_llm(RETRIEVAL_MODEL)


def get_interpretation_llm() -> ChatOpenAI:
    return _build_llm(INTERPRETATION_MODEL)
