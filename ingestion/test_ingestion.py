"""Tests for the ingestion module."""

import os
from pathlib import Path

import openpyxl
import pytest

from ingestion.ingest import chunk_schema, embed_and_store, parse_schema

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.xlsx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_xlsx(tmp_path: Path, sheets: dict) -> str:
    """Write a .xlsx file inside pytest's tmp_path from a dict of {sheet_name: [row_dicts]}.

    Returns the file path. pytest handles cleanup automatically.
    """
    path = str(tmp_path / "schema.xlsx")

    wb = openpyxl.Workbook()
    first = True
    for sheet_name, rows in sheets.items():
        if first:
            ws = wb.active
            ws.title = sheet_name
            first = False
        else:
            ws = wb.create_sheet(sheet_name)

        if rows:
            ws.append(list(rows[0].keys()))
            for row in rows:
                ws.append(list(row.values()))

    wb.save(path)
    return path


# ---------------------------------------------------------------------------
# parse_schema tests
# ---------------------------------------------------------------------------


class TestParseSchema:
    def test_reads_all_sheets(self):
        """parse_schema should return one entry per sheet."""
        parsed = parse_schema(SCHEMA_PATH)
        assert len(parsed) == 2

    def test_table_names_match_sheets(self):
        """Table names should equal the sheet names in the workbook."""
        import openpyxl as _xl

        wb = _xl.load_workbook(SCHEMA_PATH)
        parsed = parse_schema(SCHEMA_PATH)
        assert [t["table_name"] for t in parsed] == wb.sheetnames

    def test_each_table_has_required_keys(self):
        """Every entry must have table_name and columns keys."""
        for table in parse_schema(SCHEMA_PATH):
            assert "table_name" in table
            assert "columns" in table

    def test_each_column_has_required_keys(self):
        """Every column entry must have the four expected keys."""
        for table in parse_schema(SCHEMA_PATH):
            for col in table["columns"]:
                assert "column_name" in col
                assert "data_type" in col
                assert "nullable" in col
                assert "description" in col

    def test_columns_are_not_empty(self):
        """Each table should have at least one column."""
        for table in parse_schema(SCHEMA_PATH):
            assert len(table["columns"]) > 0

    def test_custom_xlsx_two_tables(self, tmp_path):
        """parse_schema works on an arbitrary two-sheet workbook."""
        path = _make_xlsx(
            tmp_path,
            {
                "VENDORS": [
                    {
                        "column_name": "VENDOR_ID",
                        "data_type": "NUMBER",
                        "nullable": "N",
                        "description": "PK",
                    }
                ],
                "INVOICES": [
                    {
                        "column_name": "INVOICE_ID",
                        "data_type": "NUMBER",
                        "nullable": "N",
                        "description": "PK",
                    }
                ],
            },
        )
        result = parse_schema(path)
        assert len(result) == 2
        assert result[0]["table_name"] == "VENDORS"
        assert result[1]["table_name"] == "INVOICES"


# ---------------------------------------------------------------------------
# Edge-case: header-only sheet (no data rows)
# ---------------------------------------------------------------------------


class TestParseSchemaEdgeCases:
    def test_header_only_sheet_returns_empty_columns(self, tmp_path):
        """A sheet with only a header row should yield an entry with no columns."""
        path = _make_xlsx(tmp_path, {"EMPTY_TABLE": []})
        result = parse_schema(path)
        assert len(result) == 1
        assert result[0]["table_name"] == "EMPTY_TABLE"
        assert result[0]["columns"] == []

    def test_column_with_empty_description(self, tmp_path):
        """A column whose description is blank should be parsed without error."""
        path = _make_xlsx(
            tmp_path,
            {
                "T": [
                    {
                        "column_name": "COL1",
                        "data_type": "VARCHAR2(10)",
                        "nullable": "Y",
                        "description": "",
                    }
                ]
            },
        )
        result = parse_schema(path)
        assert result[0]["columns"][0]["description"] == ""


# ---------------------------------------------------------------------------
# chunk_schema tests
# ---------------------------------------------------------------------------


class TestChunkSchema:
    def test_one_chunk_per_table(self):
        """chunk_schema produces exactly one chunk per non-empty table."""
        parsed = parse_schema(SCHEMA_PATH)
        chunks = chunk_schema(parsed)
        assert len(chunks) == len(parsed)

    def test_chunk_has_required_keys(self):
        """Every chunk must have table_name and chunk_text keys."""
        for chunk in chunk_schema(parse_schema(SCHEMA_PATH)):
            assert "table_name" in chunk
            assert "chunk_text" in chunk

    def test_no_empty_chunks(self):
        """No chunk should have an empty chunk_text."""
        for chunk in chunk_schema(parse_schema(SCHEMA_PATH)):
            assert chunk["chunk_text"].strip() != ""

    def test_respects_500_token_limit(self):
        """Chunks should never exceed 500 whitespace-delimited tokens."""
        for chunk in chunk_schema(parse_schema(SCHEMA_PATH)):
            assert len(chunk["chunk_text"].split()) <= 500

    def test_chunk_contains_table_name(self):
        """chunk_text should mention the table name."""
        for chunk in chunk_schema(parse_schema(SCHEMA_PATH)):
            assert chunk["table_name"] in chunk["chunk_text"]

    def test_long_table_truncated(self, tmp_path):
        """A table with many columns should be truncated to ≤500 tokens."""
        rows = [
            {
                "column_name": f"COL_{i}",
                "data_type": "VARCHAR2(255)",
                "nullable": "Y",
                "description": "A " * 20,  # inflates token count
            }
            for i in range(50)
        ]
        path = _make_xlsx(tmp_path, {"BIG_TABLE": rows})
        parsed = parse_schema(path)
        chunks = chunk_schema(parsed)
        assert len(chunks[0]["chunk_text"].split()) <= 500

    def test_header_only_table_excluded(self, tmp_path):
        """Tables with no columns should produce no chunk."""
        path = _make_xlsx(tmp_path, {"EMPTY": []})
        parsed = parse_schema(path)
        chunks = chunk_schema(parsed)
        assert chunks == []

    def test_empty_description_shown_as_no_description(self, tmp_path):
        """Columns with blank descriptions should show 'No description'."""
        path = _make_xlsx(
            tmp_path,
            {
                "T": [
                    {
                        "column_name": "C",
                        "data_type": "NUMBER",
                        "nullable": "N",
                        "description": "",
                    }
                ]
            },
        )
        parsed = parse_schema(path)
        chunks = chunk_schema(parsed)
        assert "No description" in chunks[0]["chunk_text"]


# ---------------------------------------------------------------------------
# embed_and_store tests
# ---------------------------------------------------------------------------


class TestEmbedAndStore:
    def test_collection_created_with_correct_entry_count(self, tmp_path):
        """ChromaDB collection should have one entry per chunk."""
        import chromadb

        parsed = parse_schema(SCHEMA_PATH)
        chunks = chunk_schema(parsed)
        chroma_dir = str(tmp_path / "chroma_db")

        embed_and_store(chunks, chroma_path=chroma_dir)

        client = chromadb.PersistentClient(path=chroma_dir)
        col = client.get_collection("oracle_ap_schema")
        assert col.count() == len(chunks)

    def test_collection_metadata_has_table_name(self, tmp_path):
        """Each stored entry should carry a table_name metadata field."""
        import chromadb

        parsed = parse_schema(SCHEMA_PATH)
        chunks = chunk_schema(parsed)
        chroma_dir = str(tmp_path / "chroma_db")

        embed_and_store(chunks, chroma_path=chroma_dir)

        client = chromadb.PersistentClient(path=chroma_dir)
        col = client.get_collection("oracle_ap_schema")
        results = col.get(include=["metadatas"])
        for meta in results["metadatas"]:
            assert "table_name" in meta

    def test_embed_and_store_idempotent(self, tmp_path):
        """Running embed_and_store twice should not duplicate entries."""
        import chromadb

        parsed = parse_schema(SCHEMA_PATH)
        chunks = chunk_schema(parsed)
        chroma_dir = str(tmp_path / "chroma_db")

        embed_and_store(chunks, chroma_path=chroma_dir)
        embed_and_store(chunks, chroma_path=chroma_dir)

        client = chromadb.PersistentClient(path=chroma_dir)
        col = client.get_collection("oracle_ap_schema")
        assert col.count() == len(chunks)
