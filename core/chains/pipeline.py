def _format_sql_result(raw_result: str) -> str:
    """Make ambiguous SQL Result values unambiguous before they reach the
    interpretation model - both have been observed to make models claim they
    lack database access entirely, instead of reporting what actually happened.
    """
    if not raw_result or not raw_result.strip():
        return "(The query ran successfully and returned 0 rows - there is no matching data for this question.)"
    if raw_result.strip().startswith("Error:"):
        return (
            "(The query failed to execute due to a SQL error - tell the user their "
            "question couldn't be answered because of a query error, do not guess "
            f"an answer. Raw error for debugging only, do not quote it verbatim: {raw_result})"
        )
    return raw_result


def answer_question(retrieval_chain, interpretation_chain, question: str, messages) -> dict:
    """Run retrieval then interpretation as two explicit steps, so the generated
    SQL is available to log even though the final answer comes from a second call.
    """
    retrieval_result = retrieval_chain.invoke(
        {"question": question, "top_k": 3, "messages": messages}
    )

    answer = interpretation_chain.invoke(
        {
            "question": question,
            "query": retrieval_result["query"],
            "result": _format_sql_result(retrieval_result["result"]),
            "messages": messages,
        }
    )

    return {"answer": answer, "generated_sql": retrieval_result["query"]}
