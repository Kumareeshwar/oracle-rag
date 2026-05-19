"""SQL generator module for Oracle RAG system.

Builds prompts from schema context and generates valid Oracle SQL
via the Groq API (LLaMA 3.3 70B).
"""

import os
from typing import List

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def build_prompt(query: str, context_chunks: List[str]) -> str:
    """Combine the natural language query and schema chunks into a prompt.

    The prompt instructs the model to return only valid Oracle SQL.

    Args:
        query: Natural language question from the user.
        context_chunks: List of Oracle AP schema text chunks from retrieval.

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    context = "\n".join(context_chunks) if context_chunks else "(no schema context available)"
    prompt = (
        "You are an Oracle SQL expert. Given the following Oracle AP schema context, "
        "generate a valid Oracle SQL query that answers the user's question.\n"
        "Return ONLY the SQL query — no explanation, no markdown, no extra text.\n\n"
        f"Schema context:\n{context}\n\n"
        f"User question: {query}\n\n"
        "SQL query:"
    )
    return prompt


def generate_sql(query: str, context_chunks: List[str]) -> str:
    """Generate an Oracle SQL query from a natural language question.

    Builds a prompt using build_prompt, then sends it to the Groq API
    using the LLaMA 3.3 70B model.

    Args:
        query: Natural language question from the user.
        context_chunks: List of schema text chunks from retrieval.

    Returns:
        Generated SQL string from the model. May be empty if the model
        returns no content.
    """
    prompt = build_prompt(query, context_chunks)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


def validate_sql(sql: str) -> bool:
    """Check that the SQL string starts with a recognised DML/DQL keyword.

    Args:
        sql: SQL string to validate.

    Returns:
        True if sql starts with SELECT, INSERT, UPDATE, or DELETE; False otherwise.
    """
    if not sql or not sql.strip():
        return False
    sql_upper = sql.strip().upper()
    valid_keywords = ("SELECT", "INSERT", "UPDATE", "DELETE")
    return any(sql_upper.startswith(kw) for kw in valid_keywords)
