# Task: Core Logic Module

## Your Job
Query the ChromaDB collection built by the ingestion module to retrieve relevant
schema context, then use Groq to generate a valid SQL query from a natural language input.

## Input
- ChromaDB collection "oracle_ap_schema" located at ingestion/chroma_db/
- Natural language query string from the user

## What to Build
core/retrieval.py with these functions:

- load_collection()
  → connects to ChromaDB at ingestion/chroma_db/
  → returns the "oracle_ap_schema" collection

- retrieve_context(query, collection, top_k=3)
  → embeds the query using sentence-transformers (all-MiniLM-L6-v2)
  → searches ChromaDB collection for top_k most relevant schema chunks
  → returns list of matching chunk texts

core/generator.py with these functions:

- build_prompt(query, context_chunks)
  → combines the NL query and retrieved schema chunks into a prompt
  → prompt should instruct the model to generate only valid Oracle SQL
  → returns prompt string

- generate_sql(query, context_chunks)
  → calls build_prompt to get the prompt
  → sends prompt to Groq API using LLaMA 3.3 70B model
  → returns generated SQL string

- validate_sql(sql)
  → checks that the output starts with SELECT, INSERT, UPDATE, or DELETE
  → returns True if valid, False otherwise

## Environment Variables (in .env)
- GROQ_API_KEY

## Dependencies
chromadb, sentence-transformers, groq, python-dotenv, langchain

## Tests (core/test_core.py)
Write pytest tests covering:
- load_collection connects successfully and returns a collection object
- retrieve_context returns correct number of results (top_k)
- retrieve_context returns non-empty strings
- build_prompt includes both the query and context in output
- validate_sql returns True for valid SQL strings
- validate_sql returns False for invalid or empty strings
- edge case: empty context chunks passed to build_prompt
- edge case: Groq returns empty response

## Done When
All pytest tests pass. Do not commit until tests pass.