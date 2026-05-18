# Task: Ingestion Module

## Your Job
Read Oracle schema from an Excel file where each sheet represents one table.
Chunk the schema by table, embed using a sentence transformer, and store in ChromaDB.

## Input
- File: ingestion/schema.xlsx
- Each sheet name = table name (e.g. EMPLOYEES, DEPARTMENTS)
- Columns in each sheet: column_name, data_type, nullable, description

## What to Build
ingestion/ingest.py with these functions:

- parse_schema(file_path) 
  → reads all sheets, returns list of dicts with keys: table_name, columns
  → each column entry has: column_name, data_type, nullable, description

- chunk_schema(parsed_schema)
  → converts each table into a text chunk describing the table and its columns
  → max 500 tokens per chunk
  → returns list of dicts: {table_name, chunk_text}

- embed_and_store(chunks)
  → embeds each chunk using sentence-transformers (all-MiniLM-L6-v2)
  → stores in ChromaDB collection named "oracle_ap_schema"
  → ChromaDB stored locally at ingestion/chroma_db/
  → each entry stores: id, document (chunk_text), metadata (table_name)

- main()
  → calls all three functions in sequence

## Dependencies
pandas, openpyxl, sentence-transformers, chromadb, python-dotenv

## Tests (ingestion/test_ingestion.py)
Write pytest tests covering:
- parse_schema reads all sheets and returns correct structure
- each parsed table has correct keys
- chunk_schema respects 500 token limit
- no empty chunks produced
- embed_and_store creates ChromaDB collection with correct number of entries
- edge case: sheet with only one row (header only, no data)
- edge case: column with empty description

## Done When
All pytest tests pass. Do not commit until tests pass.