# Oracle RAG System

## Project Overview
Natural language to SQL system over Oracle AP module schemas.
User types a question in plain English, system returns the correct SQL query.

## Tech Stack
- Python, LangChain, ChromaDB, Groq (llama-3.3-70b-versatile)
- FastAPI (backend API)
- React (frontend UI)
- pytest (testing)

## Architecture
- ingestion/   → parse Oracle AP schema DDLs, chunk by table, embed into ChromaDB
- core/        → semantic retrieval from ChromaDB + SQL generation via Groq
- api/         → FastAPI layer connecting core logic to frontend
- ui/          → React frontend for user to type NL queries and see SQL output
- tests/       → integration tests for full pipeline

## Rules for Agents
- Never hardcode credentials. Use .env and python-dotenv always.
- Every module must have a corresponding test file.
- All functions must have docstrings.
- Validate SQL output before returning to user.
- Keep functions small and single purpose.
- After finishing a module run the tests and fix until all pass.
