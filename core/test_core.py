"""Pytest tests for the core retrieval and SQL generation modules."""

import uuid
import numpy as np
import pytest
import chromadb
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_in_memory_collection(n_docs: int = 5):
    """Return an in-memory ChromaDB collection pre-populated with dummy docs."""
    client = chromadb.EphemeralClient()
    unique_name = f"test_collection_{uuid.uuid4().hex}"
    collection = client.create_collection(
        name=unique_name,
        metadata={"hnsw:space": "cosine"},
    )
    dim = 384
    collection.add(
        ids=[f"id{i}" for i in range(n_docs)],
        documents=[f"Oracle AP schema chunk {i}" for i in range(n_docs)],
        embeddings=[[float(i) / (n_docs * 10)] * dim for i in range(n_docs)],
    )
    return collection


# ---------------------------------------------------------------------------
# retrieval.load_collection
# ---------------------------------------------------------------------------

class TestLoadCollection:
    def test_returns_collection_object(self):
        """load_collection should return a ChromaDB Collection."""
        mock_collection = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection

        with patch("core.retrieval.chromadb.PersistentClient", return_value=mock_client_instance):
            from core.retrieval import load_collection
            result = load_collection()

        assert result is mock_collection

    def test_connects_to_correct_path(self):
        """load_collection should use the ingestion/chroma_db path."""
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = MagicMock()

        with patch("core.retrieval.chromadb.PersistentClient", return_value=mock_client_instance) as mock_client:
            from core.retrieval import load_collection
            load_collection()

        mock_client.assert_called_once_with(path="ingestion/chroma_db")

    def test_requests_correct_collection_name(self):
        """load_collection should request the oracle_ap_schema collection."""
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = MagicMock()

        with patch("core.retrieval.chromadb.PersistentClient", return_value=mock_client_instance):
            from core.retrieval import load_collection
            load_collection()

        mock_client_instance.get_or_create_collection.assert_called_once_with("oracle_ap_schema")


# ---------------------------------------------------------------------------
# retrieval.retrieve_context
# ---------------------------------------------------------------------------

class TestRetrieveContext:
    def _mock_encoder(self, dim: int = 384):
        """Return a mock SentenceTransformer that encodes to a fixed vector."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(dim, dtype=np.float32)
        return mock_model

    def test_returns_correct_number_of_results(self):
        """retrieve_context should return exactly top_k documents."""
        collection = _make_in_memory_collection(n_docs=5)

        with patch("core.retrieval.SentenceTransformer", return_value=self._mock_encoder()):
            from core.retrieval import retrieve_context
            results = retrieve_context("get invoices", collection, top_k=3)

        assert len(results) == 3

    def test_returns_non_empty_strings(self):
        """Every result from retrieve_context should be a non-empty string."""
        collection = _make_in_memory_collection(n_docs=4)

        with patch("core.retrieval.SentenceTransformer", return_value=self._mock_encoder()):
            from core.retrieval import retrieve_context
            results = retrieve_context("vendor payments", collection, top_k=2)

        for chunk in results:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_top_k_one_returns_single_result(self):
        """retrieve_context with top_k=1 should return exactly one result."""
        collection = _make_in_memory_collection(n_docs=3)

        with patch("core.retrieval.SentenceTransformer", return_value=self._mock_encoder()):
            from core.retrieval import retrieve_context
            results = retrieve_context("invoice amount", collection, top_k=1)

        assert len(results) == 1


# ---------------------------------------------------------------------------
# generator.build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_includes_query_in_output(self):
        """build_prompt result must contain the user query."""
        from core.generator import build_prompt
        query = "How many invoices were paid last month?"
        prompt = build_prompt(query, ["TABLE ap_invoices (invoice_id NUMBER)"])
        assert query in prompt

    def test_includes_context_in_output(self):
        """build_prompt result must contain the provided context chunk."""
        from core.generator import build_prompt
        chunk = "TABLE ap_invoices (invoice_id NUMBER, amount NUMBER)"
        prompt = build_prompt("total invoice amount", [chunk])
        assert chunk in prompt

    def test_includes_all_context_chunks(self):
        """All context chunks must appear in the built prompt."""
        from core.generator import build_prompt
        chunks = ["TABLE ap_invoices", "TABLE ap_vendors", "TABLE ap_payments"]
        prompt = build_prompt("find vendor invoices", chunks)
        for chunk in chunks:
            assert chunk in prompt

    def test_empty_context_chunks(self):
        """build_prompt should handle an empty context list without raising."""
        from core.generator import build_prompt
        query = "show all vendors"
        prompt = build_prompt(query, [])
        assert query in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# generator.validate_sql
# ---------------------------------------------------------------------------

class TestValidateSql:
    @pytest.mark.parametrize("sql", [
        "SELECT * FROM ap_invoices",
        "select invoice_id from ap_invoices where amount > 100",
        "INSERT INTO ap_invoices (id) VALUES (1)",
        "UPDATE ap_invoices SET amount = 200 WHERE id = 1",
        "DELETE FROM ap_invoices WHERE id = 99",
        "  SELECT id FROM ap_vendors  ",   # leading/trailing whitespace
    ])
    def test_returns_true_for_valid_sql(self, sql):
        """validate_sql should return True for valid DML/DQL statements."""
        from core.generator import validate_sql
        assert validate_sql(sql) is True

    @pytest.mark.parametrize("sql", [
        "",
        "   ",
        None,
        "DROP TABLE ap_invoices",
        "TRUNCATE TABLE ap_invoices",
        "This is not SQL at all",
        "CREATE TABLE foo (id NUMBER)",
    ])
    def test_returns_false_for_invalid_sql(self, sql):
        """validate_sql should return False for non-DML/DQL or empty input."""
        from core.generator import validate_sql
        assert validate_sql(sql) is False


# ---------------------------------------------------------------------------
# generator.generate_sql — edge case: Groq returns empty response
# ---------------------------------------------------------------------------

class TestGenerateSql:
    def test_groq_empty_response_returns_empty_string(self):
        """generate_sql should return '' when Groq returns no content."""
        mock_message = MagicMock()
        mock_message.content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.return_value = mock_response

        with patch("core.generator.Groq", return_value=mock_groq_client):
            from core.generator import generate_sql
            result = generate_sql("How many invoices?", ["TABLE ap_invoices"])

        assert result == ""

    def test_groq_returns_sql_string(self):
        """generate_sql should return the trimmed SQL from Groq's response."""
        sql = "SELECT COUNT(*) FROM ap_invoices"
        mock_message = MagicMock()
        mock_message.content = f"  {sql}  "

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.return_value = mock_response

        with patch("core.generator.Groq", return_value=mock_groq_client):
            from core.generator import generate_sql
            result = generate_sql("count invoices", ["TABLE ap_invoices"])

        assert result == sql
