import os
import re
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pypdf
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_assistant_agents")

# Global Cache for embedding model to avoid reloading on every request
_embedding_model: Optional[SentenceTransformer] = None

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        # Load small, efficient local embedding model
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("SentenceTransformer model loaded successfully.")
    return _embedding_model

class IngestionAgent:
    """
    Ingestion Agent
    Extracts text, tables, and image metadata from uploaded PDFs using pdfplumber + pypdf.
    """
    def __init__(self):
        pass

    def _table_to_text(self, table: List[List[Any]]) -> str:
        """Convert a pdfplumber table (list of rows) into a readable text block."""
        if not table:
            return ""
        lines = []
        for row in table:
            cells = [str(c).strip() if c else "" for c in row]
            lines.append(" | ".join(cells))
        return "\n".join(lines)

    def run(self, pdf_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        logger.info(f"Ingestion Agent: Starting enhanced PDF parsing for {pdf_path}...")
        pages_content = []
        metadata = {
            "filename": os.path.basename(pdf_path),
            "title": "Unknown Title",
            "author": "Unknown Author",
            "subject": "Unknown Subject",
            "pages": 0,
            "char_count": 0,
            "word_count": 0,
            "tables_found": 0,
            "images_found": 0
        }

        try:
            # Use pypdf to get metadata and page count
            reader = pypdf.PdfReader(pdf_path)
            total_pages = len(reader.pages)
            metadata["pages"] = total_pages

            # Extract PDF metadata fields
            info = reader.metadata
            if info:
                if info.title:
                    metadata["title"] = info.title
                if info.author:
                    metadata["author"] = info.author
                if info.subject:
                    metadata["subject"] = info.subject

            # Derive title from filename if still unknown
            if metadata["title"] == "Unknown Title":
                metadata["title"] = os.path.splitext(os.path.basename(pdf_path))[0].replace("_", " ").replace("-", " ").title()

            total_chars = 0
            total_words = 0
            total_tables = 0
            total_images = 0

            # Use pdfplumber for rich text + table + image extraction
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, plumber_page in enumerate(pdf.pages):
                    page_text_parts = []

                    # 1. Extract main page text (pdfplumber handles complex layouts better)
                    raw_text = plumber_page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                    # If pdfplumber text is sparse, fall back to pypdf
                    if len(raw_text.strip()) < 50:
                        fallback_text = reader.pages[page_num].extract_text() or ""
                        if len(fallback_text.strip()) > len(raw_text.strip()):
                            raw_text = fallback_text

                    text_clean = re.sub(r'\s+', ' ', raw_text).strip()
                    if text_clean:
                        page_text_parts.append(text_clean)

                    # 2. Extract tables and format them as structured text
                    try:
                        tables = plumber_page.extract_tables()
                        if tables:
                            for table_idx, table in enumerate(tables):
                                if table and any(any(c for c in row if c) for row in table):
                                    table_text = self._table_to_text(table)
                                    page_text_parts.append(f"[TABLE {table_idx + 1} on Page {page_num + 1}]:\n{table_text}")
                                    total_tables += 1
                    except Exception as table_err:
                        logger.warning(f"Table extraction error on page {page_num + 1}: {table_err}")

                    # 3. Count images (xobjects classified as images)
                    try:
                        images_on_page = plumber_page.images
                        if images_on_page:
                            img_count = len(images_on_page)
                            total_images += img_count
                            page_text_parts.append(f"[Page {page_num + 1} contains {img_count} image(s)/figure(s)]")
                    except Exception as img_err:
                        logger.warning(f"Image metadata extraction error on page {page_num + 1}: {img_err}")

                    full_page_content = "\n".join(page_text_parts)
                    char_count = len(full_page_content)
                    word_count = len(full_page_content.split())

                    total_chars += char_count
                    total_words += word_count

                    pages_content.append({
                        "page_number": page_num + 1,
                        "content": full_page_content,
                        "char_count": char_count,
                        "word_count": word_count
                    })

            metadata["char_count"] = total_chars
            metadata["word_count"] = total_words
            metadata["tables_found"] = total_tables
            metadata["images_found"] = total_images

            logger.info(
                f"Ingestion Agent: Completed. Pages: {total_pages}, Words: {total_words}, "
                f"Tables: {total_tables}, Images: {total_images}"
            )
            return pages_content, metadata

        except Exception as e:
            logger.error(f"Ingestion Agent Error: {e}")
            raise Exception(f"Failed to ingest PDF: {str(e)}")



class IndexingAgent:
    """
    Indexing Agent
    Creates text chunks, generates embeddings, and indexes them in FAISS.
    """
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for page in pages:
            text = page["content"]
            page_num = page["page_number"]
            
            if not text:
                continue

            # Standard chunking with overlap
            start = 0
            text_len = len(text)
            
            while start < text_len:
                end = start + self.chunk_size
                # If we are not at the start, check if we can adjust to a space/newline
                if end < text_len:
                    # Look back up to 50 chars for a space
                    space_idx = text.rfind(' ', end - 50, end)
                    if space_idx != -1:
                        end = space_idx
                
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num,
                        "start_char": start,
                        "end_char": start + len(chunk_text)
                    })
                
                start = end - self.chunk_overlap
                if start >= text_len - 50: # Avoid tiny remaining fragments
                    break
        return chunks

    def run(self, pages: List[Dict[str, Any]]) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        logger.info("Indexing Agent: Splitting text into chunks...")
        chunks = self._split_text(pages)
        logger.info(f"Indexing Agent: Generated {len(chunks)} chunks.")

        if not chunks:
            raise ValueError("No text content found in PDF to index.")

        logger.info("Indexing Agent: Generating embeddings with SentenceTransformer...")
        embed_model = get_embedding_model()
        chunk_texts = [c["text"] for c in chunks]
        
        # Generate embeddings
        embeddings = embed_model.encode(chunk_texts, convert_to_numpy=True)
        dimension = embeddings.shape[1]

        # Use Inner Product (cosine similarity after normalisation)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        logger.info(f"Indexing Agent: FAISS Index created with {index.ntotal} vectors.")
        return index, chunks


class RetrieverAgent:
    """
    Retriever Agent
    Finds relevant sections based on user queries from the active FAISS index.
    """
    def __init__(self, index: faiss.Index, chunks: List[Dict[str, Any]]):
        self.index = index
        self.chunks = chunks

    def run(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        logger.info(f"Retriever Agent: Searching for query: '{query}'...")
        if not self.index or self.index.ntotal == 0:
            logger.warning("Retriever Agent: Empty or missing FAISS index.")
            return []

        embed_model = get_embedding_model()
        query_vector = embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)

        # Search FAISS index
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            
            chunk = self.chunks[idx]
            # Convert inner product score back to typical range or similarity confidence
            similarity = float(score)
            
            results.append({
                "text": chunk["text"],
                "page": chunk["page"],
                "score": round(similarity, 4)
            })

        logger.info(f"Retriever Agent: Retrieved {len(results)} relevant chunks.")
        return results


class SummarizerAgent:
    """
    Summarizer Agent
    Uses Groq LLM to generate a response based on retrieved text.
    """
    def __init__(self, api_key: str, model_name: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def run(self, query: str, retrieved_chunks: List[Dict[str, Any]], history: List[Dict[str, str]] = None) -> str:
        logger.info(f"Summarizer Agent: Generating answer using {self.model_name}...")
        
        # Compile retrieved chunks into a context string
        context_str = ""
        for i, chunk in enumerate(retrieved_chunks):
            context_str += f"[Source {i+1} - Page {chunk['page']}]: {chunk['text']}\n\n"

        system_prompt = (
            "You are a helpful, expert Summarizer Agent for a PDF Research Assistant. "
            "Your job is to answer the user's question based strictly and ONLY on the provided Context. "
            "The context may contain plain text, structured table data (formatted as pipe-separated rows prefixed with [TABLE ...]), "
            "and notes about images or figures (prefixed with [Page X contains N image(s)/figure(s)]).\n\n"
            "Follow these rules:\n"
            "1. Be direct, clear, and comprehensive. When tables are in context, present data in a formatted markdown table.\n"
            "2. If context mentions images or figures, acknowledge them and describe any associated text labels or captions.\n"
            "3. Cite your sources clearly using '[Page X]' or '[Source Y]' based on the Page numbers in the context.\n"
            "4. If the answer cannot be found in the provided context, state clearly: 'I cannot find the answer in the provided document.' "
            "Do NOT make up any facts or extrapolate beyond the text.\n"
            "5. Keep your tone neutral, scientific, and professional."
        )

        user_content = f"Context:\n{context_str}\n\nQuestion: {query}"

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history for continuity if available
        if history:
            # Only add last 4 messages to avoid context overflow and keep focus on current query
            for msg in history[-4:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_content})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=1024
            )
            answer = response.choices[0].message.content
            logger.info("Summarizer Agent: Answer generated successfully.")
            return answer
        except Exception as e:
            logger.error(f"Summarizer Agent Error: {e}")
            raise Exception(f"Summarizer failed: {str(e)}")


class ValidatorAgent:
    """
    Validator Agent
    Cross-checks the generated answer against source text to detect hallucinations and grade support.
    """
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def run(self, query: str, retrieved_chunks: List[Dict[str, Any]], generated_answer: str) -> Dict[str, Any]:
        logger.info(f"Validator Agent: Verifying answer using {self.model_name}...")

        context_str = ""
        for i, chunk in enumerate(retrieved_chunks):
            context_str += f"[Source {i+1} - Page {chunk['page']}]: {chunk['text']}\n\n"

        system_prompt = (
            "You are a strict, objective Validator Agent (LLM-as-a-Judge) for a PDF Research Assistant.\n"
            "Your task is to cross-check the proposed 'Answer' against the 'Source Context' and verify its factual alignment.\n"
            "You must return a JSON object with the following fields:\n"
            "1. 'score': an integer from 0 to 100 representing the factual alignment of the answer with the context. "
            "100 means everything is fully supported; 0 means nothing is supported or it directly contradicts.\n"
            "2. 'claims': a list of objects representing key claims made in the answer. Each object must have:\n"
            "   - 'claim': a short description of the statement.\n"
            "   - 'status': either 'fully_supported', 'partially_supported', or 'unsupported'.\n"
            "   - 'page_citation': page numbers backing this claim, or null.\n"
            "   - 'reason': a brief explanation of why it has this status.\n"
            "3. 'hallucinations': a list of strings detailing any facts, assertions, or assumptions in the answer that are not supported by the context.\n"
            "4. 'verdict': a string either 'PASS' (score >= 80), 'WARNING' (score 50-79), or 'FAIL' (score < 50).\n\n"
            "Only evaluate based on the provided Source Context. Do not bring in external knowledge.\n"
            "Format your response as a valid JSON object ONLY. Do not enclose it in markdown code blocks."
        )

        user_content = (
            f"Source Context:\n{context_str}\n\n"
            f"User Question: {query}\n\n"
            f"Proposed Answer:\n{generated_answer}"
        )

        try:
            # We enforce JSON mode
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            raw_response = response.choices[0].message.content.strip()
            # Parse the response
            validation_data = json.loads(raw_response)
            logger.info(f"Validator Agent: Evaluation completed. Score: {validation_data.get('score', 0)}%, Verdict: {validation_data.get('verdict', 'FAIL')}")
            return validation_data
        except Exception as e:
            logger.error(f"Validator Agent Error: {e}")
            # Fallback evaluation structure if JSON parse fails
            return {
                "score": 50,
                "claims": [{"claim": "Auto-generated verification failed", "status": "unsupported", "page_citation": None, "reason": f"Validator failed to compile validation: {str(e)}"}],
                "hallucinations": [f"Validation system exception: {str(e)}"],
                "verdict": "WARNING"
            }


class ConversationAgent:
    """
    Conversation Agent
    Coordinates the multi-agent pipeline, manages active documents, logs steps, and delivers Q&A.
    """
    def __init__(self):
        # Maps session_id or 'default' to PDF state
        self.active_documents: Dict[str, Dict[str, Any]] = {}

    def set_document(self, session_id: str, pages: List[Dict[str, Any]], metadata: Dict[str, Any], index: faiss.Index, chunks: List[Dict[str, Any]]):
        self.active_documents[session_id] = {
            "pages": pages,
            "metadata": metadata,
            "index": index,
            "chunks": chunks
        }
        logger.info(f"Conversation Agent: Active document set for session {session_id}.")

    def get_document_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        doc = self.active_documents.get(session_id)
        return doc["metadata"] if doc else None

    def run_qa(
        self, 
        session_id: str, 
        query: str, 
        api_key: str, 
        summarizer_model: str = "llama-3.1-8b-instant",
        validator_model: str = "llama-3.3-70b-versatile",
        history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        
        logger.info(f"Conversation Agent: Starting Q&A workflow for session {session_id}...")
        
        doc = self.active_documents.get(session_id)
        if not doc:
            raise ValueError("No active document loaded. Please upload a PDF first.")

        agent_logs = []
        
        # 1. Retriever Agent
        agent_logs.append({"agent": "Retriever Agent", "status": "running", "message": "Searching document database for matching passages..."})
        retriever = RetrieverAgent(doc["index"], doc["chunks"])
        retrieved_chunks = retriever.run(query)
        
        if not retrieved_chunks:
            agent_logs.append({"agent": "Retriever Agent", "status": "completed", "message": "No relevant text chunks found."})
            return {
                "answer": "I could not find any text chunks relevant to your question in the PDF document.",
                "retrieved_chunks": [],
                "validation": {
                    "score": 100,
                    "claims": [],
                    "hallucinations": [],
                    "verdict": "PASS"
                },
                "agent_logs": agent_logs
            }

        agent_logs.append({
            "agent": "Retriever Agent", 
            "status": "completed", 
            "message": f"Retrieved {len(retrieved_chunks)} relevant sections from page(s): {', '.join(set(str(c['page']) for c in retrieved_chunks))}"
        })

        # 2. Summarizer Agent
        agent_logs.append({"agent": "Summarizer Agent", "status": "running", "message": f"Synthesizing draft response using Groq model {summarizer_model}..."})
        summarizer = SummarizerAgent(api_key, summarizer_model)
        generated_answer = summarizer.run(query, retrieved_chunks, history)
        agent_logs.append({"agent": "Summarizer Agent", "status": "completed", "message": "Synthesized draft response with citations."})

        # 3. Validator Agent
        agent_logs.append({"agent": "Validator Agent", "status": "running", "message": f"Cross-checking draft response against source texts using {validator_model}..."})
        validator = ValidatorAgent(api_key, validator_model)
        validation_report = validator.run(query, retrieved_chunks, generated_answer)
        agent_logs.append({
            "agent": "Validator Agent", 
            "status": "completed", 
            "message": f"Verification completed. Score: {validation_report.get('score', 0)}%. Verdict: {validation_report.get('verdict', 'FAIL')}"
        })

        # Final packaging
        agent_logs.append({"agent": "Conversation Agent", "status": "completed", "message": "Packaged agent metadata and returned response."})

        return {
            "answer": generated_answer,
            "retrieved_chunks": retrieved_chunks,
            "validation": validation_report,
            "agent_logs": agent_logs
        }
