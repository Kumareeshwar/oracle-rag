"""Oracle AP Schema Ingestion Module.

Parses Oracle AP schema from Excel, chunks by table, embeds using
sentence-transformers, and stores in ChromaDB.
"""

import os

import chromadb
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


def _str(value) -> str:
    """Convert a cell value to str, mapping NaN/None to empty string."""
    import math

    if value is None:
        return ""
    try:
        if math.isnan(float(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "oracle_ap_schema"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TOKENS = 500


def parse_schema(file_path: str) -> list:
    """Parse Oracle AP schema from an Excel file.

    Each sheet represents one table. The sheet name is the table name.
    Expected sheet columns: column_name, data_type, nullable, description.

    Args:
        file_path: Path to the .xlsx schema file.

    Returns:
        List of dicts, each with keys:
            - table_name (str)
            - columns (list of dicts with keys: column_name, data_type,
              nullable, description)
    """
    xl = pd.ExcelFile(file_path)
    result = []
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        columns = []
        for _, row in df.iterrows():
            col = {
                "column_name": _str(row.get("column_name")),
                "data_type": _str(row.get("data_type")),
                "nullable": _str(row.get("nullable")),
                "description": _str(row.get("description")),
            }
            columns.append(col)
        result.append({"table_name": sheet_name, "columns": columns})
    return result


def chunk_schema(parsed_schema: list) -> list:
    """Convert each parsed table into a descriptive text chunk.

    One chunk per table. Chunks exceeding MAX_TOKENS words are truncated.
    Tables with no columns produce no chunk.

    Args:
        parsed_schema: Output of parse_schema().

    Returns:
        List of dicts, each with keys:
            - table_name (str)
            - chunk_text (str)
    """
    chunks = []
    for table in parsed_schema:
        table_name = table["table_name"]
        lines = [f"Table: {table_name}"]
        for col in table["columns"]:
            nullable_label = (
                "NOT NULL" if col["nullable"].strip().upper() == "N" else "NULLABLE"
            )
            desc = col["description"].strip()
            if not desc or desc.lower() == "nan":
                desc = "No description"
            line = (
                f"  - {col['column_name']} ({col['data_type']}, "
                f"{nullable_label}): {desc}"
            )
            lines.append(line)

        chunk_text = "\n".join(lines)

        # Enforce token limit (approximated as whitespace-separated words)
        words = chunk_text.split()
        if len(words) > MAX_TOKENS:
            chunk_text = " ".join(words[:MAX_TOKENS])

        if chunk_text.strip() and len(table["columns"]) > 0:
            chunks.append({"table_name": table_name, "chunk_text": chunk_text})
    return chunks


def embed_and_store(chunks: list, chroma_path: str = CHROMA_PATH) -> None:
    """Embed each chunk and persist in a ChromaDB collection.

    Uses the all-MiniLM-L6-v2 sentence-transformer model.
    Recreates the collection on each run to keep it in sync with the schema.

    Args:
        chunks: Output of chunk_schema().
        chroma_path: Directory path for ChromaDB persistence.
            Defaults to ingestion/chroma_db/ relative to this file.
    """
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=chroma_path)

    # Recreate the collection so it stays in sync with the current schema
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk["chunk_text"]).tolist()
        collection.add(
            ids=[f"{chunk['table_name']}_{i}"],
            documents=[chunk["chunk_text"]],
            embeddings=[embedding],
            metadatas=[{"table_name": chunk["table_name"]}],
        )


def main() -> None:
    """Run the full ingestion pipeline.

    Reads schema.xlsx from the same directory, parses it, chunks it,
    embeds the chunks, and stores them in ChromaDB.
    """
    schema_path = os.path.join(os.path.dirname(__file__), "schema.xlsx")
    parsed = parse_schema(schema_path)
    chunks = chunk_schema(parsed)
    embed_and_store(chunks)
    print(
        f"Ingested {len(chunks)} chunk(s) into "
        f"ChromaDB collection '{COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    main()
