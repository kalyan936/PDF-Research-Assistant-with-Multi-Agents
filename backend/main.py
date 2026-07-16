import os
import shutil
import uuid
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.agents import IngestionAgent, IndexingAgent, ConversationAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_assistant_main")

app = FastAPI(
    title="PDF Research Assistant with Multi-Agents",
    description="Backend API for the PDF Research Assistant powered by Groq and Sentence-Transformers"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup workspace directories
WORKSPACE_DIR = os.getcwd()
TEMP_DIR = os.path.join(WORKSPACE_DIR, "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

# Instantiate global coordinator agents
conversation_agent = ConversationAgent()
ingestion_agent = IngestionAgent()
indexing_agent = IndexingAgent()

# Pydantic models for chat requests
class ChatRequest(BaseModel):
    message: str = Field(..., description="User question or chat message")
    session_id: str = Field("default", description="Session identifier")
    api_key: Optional[str] = Field(None, description="Groq API Key (falls back to backend env var if not supplied)")
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="Chat history context")
    summarizer_model: Optional[str] = Field("llama-3.1-8b-instant", description="Groq model for summarization")
    validator_model: Optional[str] = Field("llama-3.3-70b-versatile", description="Groq model for validation")

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...), session_id: str = "default"):
    """
    Upload a PDF file.
    Runs Ingestion Agent to parse, and Indexing Agent to chunk and index in FAISS.
    """
    logger.info(f"Received file upload: {file.filename} for session: {session_id}")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Create a unique path for the uploaded file in temp workspace folder
    file_id = str(uuid.uuid4())
    temp_file_path = os.path.join(TEMP_DIR, f"{file_id}_{file.filename}")

    try:
        # Save file to temp folder
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Ingestion Agent
        pages, metadata = ingestion_agent.run(temp_file_path)

        # 2. Indexing Agent
        index, chunks = indexing_agent.run(pages)

        # 3. Store in Conversation Agent session
        conversation_agent.set_document(session_id, pages, metadata, index, chunks)

        return {
            "status": "success",
            "message": "PDF ingested and indexed successfully.",
            "metadata": metadata
        }

    except ValueError as ve:
        logger.error(f"Validation error during indexing: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        # Clean up the physical file to save space and respect privacy
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as cleanup_err:
                logger.warning(f"Failed to delete temp file {temp_file_path}: {cleanup_err}")

@app.post("/api/chat")
async def chat_document(request: ChatRequest):
    """
    Chat with the active PDF.
    Coordinates Retriever, Summarizer, and Validator agents.
    """
    session_id = request.session_id
    query = request.message
    
    # Resolve Groq API Key (request body or environment variable)
    groq_api_key = request.api_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(
            status_code=401, 
            detail="Groq API Key is required. Please set it in the settings panel or backend environment."
        )

    # Check if document is indexed
    doc_meta = conversation_agent.get_document_metadata(session_id)
    if not doc_meta:
        raise HTTPException(
            status_code=400, 
            detail="No document loaded. Please upload a PDF first."
        )

    try:
        response_data = conversation_agent.run_qa(
            session_id=session_id,
            query=query,
            api_key=groq_api_key,
            summarizer_model=request.summarizer_model,
            validator_model=request.validator_model,
            history=request.history
        )
        return response_data

    except Exception as e:
        logger.error(f"QA workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status(session_id: str = "default"):
    """
    Check if a PDF is indexed and fetch its metadata.
    """
    meta = conversation_agent.get_document_metadata(session_id)
    if meta:
        return {"status": "indexed", "metadata": meta}
    return {"status": "empty", "metadata": None}

@app.post("/api/reset")
async def reset_session(session_id: str = "default"):
    """
    Reset and clear the current PDF session.
    """
    if session_id in conversation_agent.active_documents:
        del conversation_agent.active_documents[session_id]
        logger.info(f"Session {session_id} reset successfully.")
        return {"status": "success", "message": "Session reset successfully."}
    return {"status": "success", "message": "Session was already empty."}

# Mount static folder for frontend UI
static_path = os.path.join(WORKSPACE_DIR, "backend", "static")
if os.path.exists(static_path):
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
else:
    logger.warning(f"Static folder not found at {static_path}. Frontend will not be served directly by FastAPI.")
