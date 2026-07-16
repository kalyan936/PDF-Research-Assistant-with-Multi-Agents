# 🤖 PDF Research Assistant with Multi-Agents

A production-grade multi-agent AI system for intelligent PDF research powered by **Groq's ultra-fast LLMs** and **local semantic embeddings**.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![Groq](https://img.shields.io/badge/Groq-LLM-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 🎯 What It Does

Upload any PDF (research papers, legal documents, technical reports) and get:
- **Semantic Q&A** — Ask natural language questions and get citations.
- **Table & Image Understanding** — Extracts and reasons over tables and figure counts.
- **Factual Validation** — An LLM-as-a-Judge agent scores every response for hallucinations.
- **Live Agent Pipeline** — Watch 6 specialized agents collaborate in real-time.

---

## 🛠️ Architecture: 6 Specialized Agents

| Agent | Role |
|---|---|
| **Ingestion Agent** | Parses PDF text, tables, and image metadata using `pdfplumber` |
| **Indexing Agent** | Chunks text and creates a FAISS semantic vector index using `sentence-transformers` |
| **Retriever Agent** | Finds the most relevant document sections via cosine similarity search |
| **Summarizer Agent** | Synthesizes answers with source citations using a Groq LLM |
| **Validator Agent** | Cross-checks answers against source chunks (LLM-as-a-Judge) with a factual accuracy score |
| **Conversation Agent** | Coordinates all agents, manages sessions, delivers the final response |

---

## 🚀 Run Locally

### 1. Clone & install dependencies
```bash
git clone https://github.com/kalyan936/PDF-Research-Assistant-with-Multi-Agents.git
cd PDF-Research-Assistant-with-Multi-Agents
pip install -r requirements.txt
```

### 2. Start the server
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080 --reload
```

### 3. Open the app
Navigate to → **http://localhost:8080**

### 4. Configure your Groq API Key
- Click the **⚙️ Settings** button in the top right.
- Paste your [Groq API Key](https://console.groq.com).
- Upload any PDF and start asking questions!

---

## ☁️ Deploy to the Cloud (Render)

> GitHub Pages only hosts static sites. This project has a Python backend and must be deployed on a platform that supports Python.

### Deploy to Render (Free Tier)
1. Go to [render.com](https://render.com) → **New Web Service**
2. Connect your GitHub repository.
3. Set the following:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable: `GROQ_API_KEY` = your key (optional — users can also supply it via the UI).
5. Deploy 🚀

---

## 📁 Project Structure

```
PDF-Research-Assistant-with-Multi-Agents/
├── backend/
│   ├── agents.py          # All 6 agent classes
│   ├── main.py            # FastAPI server & API endpoints
│   ├── requirements.txt   # Backend dependencies
│   └── static/
│       ├── index.html     # Frontend UI
│       ├── style.css      # Dark cyberpunk styles
│       └── app.js         # Frontend logic & pipeline animation
├── test_pipeline.py       # Automated test suite (7 tests)
├── requirements.txt       # Root dependencies (for Render)
├── Procfile               # Process file for Heroku/Render/Railway
└── README.md
```

---

## 🔑 Tech Stack

| Layer | Technology |
|---|---|
| **LLM Provider** | [Groq API](https://groq.com) (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (runs locally) |
| **Vector Search** | `faiss-cpu` — fast approximate nearest neighbor |
| **PDF Parsing** | `pdfplumber` + `pypdf` — text, tables, and images |
| **Backend** | Python `FastAPI` + `uvicorn` |
| **Frontend** | Vanilla HTML/CSS/JS — no framework needed |

---

## 📜 License

MIT License — free to use, modify, and distribute.
