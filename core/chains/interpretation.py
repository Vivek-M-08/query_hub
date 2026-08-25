from langchain_core.output_parsers import StrOutputParser

from core.llm.client import get_interpretation_llm
from prompts import mitra_answer_prompt


def build_interpretation_chain():
    """MItra system prompt + retrieval result -> plain-language answer."""
    return mitra_answer_prompt | get_interpretation_llm() | StrOutputParser()
