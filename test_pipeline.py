import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import faiss
from fastapi.testclient import TestClient

# Adjust import path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.agents import IngestionAgent, IndexingAgent, RetrieverAgent, SummarizerAgent, ValidatorAgent, ConversationAgent
from backend.main import app

class TestMultiAgentPipeline(unittest.TestCase):
    
    def setUp(self):
        # Sample document chunks
        self.dummy_pages = [
            {"page_number": 1, "content": "The quick brown fox jumps over the lazy dog. Quantum mechanics is a fundamental theory in physics.", "char_count": 92, "word_count": 16},
            {"page_number": 2, "content": "Superconductivity is a set of physical properties observed in certain materials. Groq is an AI infrastructure company.", "char_count": 115, "word_count": 17}
        ]
        
    def test_indexing_agent_chunking_and_faiss(self):
        """Test that IndexingAgent splits text correctly and creates FAISS indices."""
        # Force small chunk size to verify it splits
        indexer = IndexingAgent(chunk_size=40, chunk_overlap=10)
        index, chunks = indexer.run(self.dummy_pages)
        
        # Verify FAISS index type and dimensions
        self.assertIsInstance(index, faiss.IndexFlatIP)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(index.ntotal, len(chunks))
        
        # Verify first chunk content
        self.assertIn("text", chunks[0])
        self.assertIn("page", chunks[0])
        self.assertEqual(chunks[0]["page"], 1)

    def test_retriever_agent_semantic_search(self):
        """Test RetrieverAgent finds matching passages based on embedding similarity."""
        indexer = IndexingAgent(chunk_size=100, chunk_overlap=20)
        index, chunks = indexer.run(self.dummy_pages)
        
        retriever = RetrieverAgent(index, chunks)
        # Search query matching first page context
        results = retriever.run("quantum theory", top_k=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["page"], 1)
        self.assertIn("quantum", results[0]["text"].lower())
        self.assertGreater(results[0]["score"], 0)

    @patch('backend.agents.Groq')
    def test_summarizer_agent(self, mock_groq_class):
        """Test SummarizerAgent formats prompt and parses Groq completions."""
        # Mock Groq client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Mocked summary based on quantum physics context [Page 1]."))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_client
        
        summarizer = SummarizerAgent(api_key="mock_key", model_name="llama-3.1-8b-instant")
        retrieved_chunks = [{"text": "Quantum mechanics rules the micro world.", "page": 1, "score": 0.9}]
        
        answer = summarizer.run("Tell me about quantum physics", retrieved_chunks)
        
        self.assertEqual(answer, "Mocked summary based on quantum physics context [Page 1].")
        # Verify model used
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "llama-3.1-8b-instant")

    @patch('backend.agents.Groq')
    def test_validator_agent(self, mock_groq_class):
        """Test ValidatorAgent parses structured JSON score results."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        
        # Mock a successful JSON validation report from Groq LLM-as-a-judge
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"score": 95, "claims": [{"claim": "Fox jumps over dog", "status": "fully_supported", "page_citation": 1, "reason": "Explicitly stated"}], "hallucinations": [], "verdict": "PASS"}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_client
        
        validator = ValidatorAgent(api_key="mock_key", model_name="llama-3.3-70b-versatile")
        chunks = [{"text": "The quick brown fox jumps over the lazy dog.", "page": 1, "score": 0.95}]
        
        report = validator.run(
            query="What does the fox do?",
            retrieved_chunks=chunks,
            generated_answer="The fox jumps over the lazy dog."
        )
        
        self.assertEqual(report["score"], 95)
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(len(report["claims"]), 1)
        self.assertEqual(report["claims"][0]["status"], "fully_supported")

class TestAPIEndpoints(unittest.TestCase):
    
    def setUp(self):
        self.client = TestClient(app)
        
    def test_api_status_empty(self):
        """Test status endpoint before upload returns empty."""
        response = self.client.get("/api/status?session_id=test_session")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "empty")

    def test_api_reset(self):
        """Test that reset endpoint works correctly."""
        response = self.client.post("/api/reset?session_id=test_session")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")

    def test_chat_unauthorized_without_key(self):
        """Test that chat endpoint demands Groq key."""
        # Run chat post request without api_key
        response = self.client.post(
            "/api/chat",
            json={
                "message": "Hello document",
                "session_id": "test_session",
                "api_key": ""
            }
        )
        # Should raise 401 Unauthorized if GROQ_API_KEY environment variable is also missing
        if not os.getenv("GROQ_API_KEY"):
            self.assertEqual(response.status_code, 401)
            self.assertIn("API Key is required", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
