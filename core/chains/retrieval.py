import re
from operator import itemgetter
from pathlib import Path

import pandas as pd
from langchain.chains import create_sql_query_chain
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config.settings import get_analytics_db_url
from core.llm.client import get_retrieval_llm

TABLE_DESC_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "table_desc_analytics_db.csv"

_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SQL_KEYWORD_RE = re.compile(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)


def clean_generated_sql(raw_sql: str) -> str:
    """Extract a bare SQL statement from an LLM's raw completion.

    create_sql_query_chain's default prompt assumes the model emits only the
    SQL text, but capable chat models (e.g. Claude via OpenRouter) commonly
    restate the question and/or wrap the query in a markdown code fence.
    Passing that raw text straight to the DB throws a syntax error - strip it.
    """
    text = raw_sql.strip()

    fence_match = _SQL_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    keyword_match = _SQL_KEYWORD_RE.search(text)
    if keyword_match:
        text = text[keyword_match.start():].strip()

    return text


def _build_contextual_question(question: str, messages) -> str:
    """Fold recent conversation into the question text before SQL generation.

    create_sql_query_chain's default prompt has no memory of prior turns at
    all - it only ever sees the current bare question. Without this, a
    follow-up like "what about Karnataka?" generates SQL with no idea what
    metric/state/time period is actually being asked about.
    """
    if not messages:
        return question

    history_lines = [
        f"{'User' if message.type == 'human' else 'Assistant'}: {message.content}"
        for message in messages[-6:]
    ]
    history_text = "\n".join(history_lines)

    return (
        "Given this recent conversation:\n"
        f"{history_text}\n\n"
        "Answer this follow-up question by first restating it as a complete, standalone "
        "question with every implicit reference resolved (same metric/filters/topic as the "
        "conversation above, but the new state/time period/etc. if one is given), then write "
        "a SQL query for that complete question exactly as rigorously as you would for a "
        "brand new question - same joins, same aggregation (COUNT/SUM/GROUP BY as appropriate), "
        "same filters on every relevant column (e.g. state). Do not write a simplified or "
        f"partial query just because this is a follow-up.\n\nFollow-up question: {question}"
    )


class Table(BaseModel):
    name: str = Field(description="Name of table in SQL database.")


def get_table_details() -> str:
    """Flatten the analytics_db table descriptions into a text block for the LLM."""
    table_description = pd.read_csv(TABLE_DESC_PATH)
    table_details = ""
    for _, row in table_description.iterrows():
        table_details += f"Table Name: {row['Table']}\nTable Description: {row['Description']}\n\n"
    return table_details


def create_table_extraction_chain(llm, table_details: str):
    table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
    The tables are:

    {table_details}

    Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

    return {"input": itemgetter("question")} | create_extraction_chain_pydantic(
        Table, llm, system_message=table_details_prompt
    )


def get_analytics_db() -> SQLDatabase:
    return SQLDatabase.from_uri(get_analytics_db_url())


def build_retrieval_chain():
    """Table-extraction -> NL-to-SQL -> SQL execution, against analytics_db.

    Returned chain's invoke() output includes the original inputs plus
    "tables", "query" (generated SQL), and "result" (execution output).
    """
    db = get_analytics_db()
    llm = get_retrieval_llm()
    table_details = get_table_details()

    table_chain = create_table_extraction_chain(llm, table_details)
    generate_query = create_sql_query_chain(llm, db) | RunnableLambda(clean_generated_sql)
    execute_query = QuerySQLDataBaseTool(db=db)

    # Only SQL generation gets the conversation-enriched question - table
    # extraction's forced tool-calling turned out to be unreliable when fed
    # the longer, conversational text instead of a clean bare question.
    contextualize_for_sql = RunnableLambda(
        lambda x: {**x, "question": _build_contextual_question(x["question"], x.get("messages", []))}
    )

    return (
        RunnablePassthrough.assign(tables=table_chain)
        .assign(query=contextualize_for_sql | generate_query)
        .assign(result=itemgetter("query") | execute_query)
    )
