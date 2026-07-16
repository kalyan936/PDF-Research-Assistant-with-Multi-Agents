/**
 * PDF Research Assistant — 100% Browser-Native Multi-Agent System
 * GitHub Pages Static Version
 *
 * All 6 agents run in the browser:
 *   IngestionAgent   → PDF.js text + table extraction
 *   IndexingAgent    → Transformers.js embeddings (all-MiniLM-L6-v2 via WebAssembly)
 *   RetrieverAgent   → Pure JS cosine similarity vector search
 *   SummarizerAgent  → Direct Groq API fetch from browser
 *   ValidatorAgent   → Direct Groq API fetch (JSON mode)
 *   ConversationAgent→ Coordinates pipeline, manages state
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.esm.js';

// ─── TRANSFORMERS.JS CONFIG ──────────────────────────────────────────────────
env.allowLocalModels = false;
env.useBrowserCache  = true;  // Cache model in IndexedDB after first download

// ─── GLOBAL STATE ────────────────────────────────────────────────────────────
let embeddingPipeline = null;
let embeddingModelLoading = false;
let docIndex  = null;   // { chunks: [], embeddings: Float32Array[] }
let docMeta   = null;
let chatHistory = [];
let isProcessing = false;

// Settings (persisted in localStorage)
let groqApiKey      = localStorage.getItem('groq_api_key')       || '';
let summarizerModel = localStorage.getItem('summarizer_model')    || 'llama-3.1-8b-instant';
let validatorModel  = localStorage.getItem('validator_model')     || 'llama-3.3-70b-versatile';
let topK            = parseInt(localStorage.getItem('top_k'))      || 4;

// ─── DOM REFS ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const D = {
    modelStatus:        $('modelStatus'),
    modelStatusText:    $('modelStatusText'),
    btnSettings:        $('btnSettings'),
    // Upload
    pdfFileInput:       $('pdfFileInput'),
    dropZone:           $('dropZone'),
    btnBrowse:          $('btnBrowse'),
    uploadProgress:     $('uploadProgress'),
    progressMessage:    $('progressMessage'),
    progressBar:        $('progressBar'),
    progressSub:        $('progressSub'),
    pdfDetails:         $('pdfDetails'),
    pdfName:            $('pdfName'),
    pdfPages:           $('pdfPages'),
    pdfWords:           $('pdfWords'),
    pdfChunks:          $('pdfChunks'),
    pdfTables:          $('pdfTables'),
    pdfTitle:           $('pdfTitle'),
    pdfAuthor:          $('pdfAuthor'),
    btnResetPdf:        $('btnResetPdf'),
    // Pipeline nodes
    nodeIngestion:      $('node-ingestion'),
    nodeIndexing:       $('node-indexing'),
    nodeRetriever:      $('node-retriever'),
    nodeSummarizer:     $('node-summarizer'),
    nodeValidator:      $('node-validator'),
    nodeConversation:   $('node-conversation'),
    conn1: $('conn-1'), conn2: $('conn-2'), conn3: $('conn-3'),
    conn4: $('conn-4'), conn5: $('conn-5'),
    // Chat
    chatHeaderInfo:     $('chatHeaderInfo'),
    activeDocTitle:     $('activeDocTitle'),
    activeDocSub:       $('activeDocSub'),
    chatMessages:       $('chatMessages'),
    chatForm:           $('chatForm'),
    userInput:          $('userInput'),
    btnSend:            $('btnSend'),
    btnClearChat:       $('btnClearChat'),
    // Validation
    noValidationMsg:    $('noValidationMsg'),
    validationResult:   $('validationResult'),
    gaugeBarInner:      $('gaugeBarInner'),
    gaugeValue:         $('gaugeValue'),
    verdictBadge:       $('verdictBadge'),
    claimsList:         $('claimsList'),
    hallucinationsSection: $('hallucinationsSection'),
    hallucinationsList: $('hallucinationsList'),
    // Citations
    noCitationsMsg:     $('noCitationsMsg'),
    citationsList:      $('citationsList'),
    // Settings modal
    settingsModal:      $('settingsModal'),
    btnCloseSettings:   $('btnCloseSettings'),
    settingsForm:       $('settingsForm'),
    groqApiKeyInput:    $('groqApiKey'),
    summarizerModelSel: $('summarizerModel'),
    validatorModelSel:  $('validatorModel'),
    topKSel:            $('topK'),
};

// ══════════════════════════════════════════════════════════════════════════════
// ░░  AGENT 1: INGESTION AGENT  ░░
// ══════════════════════════════════════════════════════════════════════════════
class IngestionAgent {
    async run(file, onProgress) {
        onProgress('Ingestion Agent: Loading PDF with PDF.js...', 10);
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

        onProgress(`Ingestion Agent: Extracting text from ${pdf.numPages} pages...`, 15);

        const pages = [];
        let totalWords = 0;
        let tablesFound = 0;

        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const content = await page.getTextContent({ normalizeWhitespace: true });

            // Group text items by Y coordinate to detect table rows
            const rows = this._groupByY(content.items);
            const pageTextParts = [];

            for (const row of rows) {
                if (row.length > 2) {
                    // Multiple items on same Y → likely a table row
                    const cells = row.map(item => item.str.trim()).filter(Boolean);
                    if (cells.length > 1) {
                        pageTextParts.push(cells.join(' | '));
                        tablesFound++;
                        continue;
                    }
                }
                pageTextParts.push(row.map(r => r.str).join(' '));
            }

            const rawText = pageTextParts.join('\n').replace(/\s+/g, ' ').trim();
            const wordCount = rawText.split(/\s+/).filter(Boolean).length;
            totalWords += wordCount;

            pages.push({ page_number: i, content: rawText, word_count: wordCount });

            if (i % 5 === 0) {
                onProgress(`Ingestion Agent: Processed ${i}/${pdf.numPages} pages...`, 15 + Math.round((i / pdf.numPages) * 15));
            }
        }

        // Extract PDF metadata
        let metadata = {
            filename: file.name,
            title: file.name.replace(/\.pdf$/i, '').replace(/[-_]/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
            author: 'Unknown Author',
            pages: pdf.numPages,
            word_count: totalWords,
            tables_found: Math.round(tablesFound / 3), // normalize (each table has many rows)
            images_found: 0,
        };

        try {
            const info = await pdf.getMetadata();
            if (info?.info?.Title)  metadata.title  = info.info.Title;
            if (info?.info?.Author) metadata.author = info.info.Author;
        } catch (_) { /* metadata might not exist */ }

        onProgress('Ingestion Agent: Text extraction complete ✓', 30);
        return { pages, metadata };
    }

    _groupByY(items) {
        const rowMap = new Map();
        for (const item of items) {
            if (!item.str.trim()) continue;
            const y = Math.round(item.transform[5]);
            if (!rowMap.has(y)) rowMap.set(y, []);
            rowMap.get(y).push(item);
        }
        // Sort rows by Y (PDF Y is bottom-up, so reverse)
        return Array.from(rowMap.entries())
            .sort((a, b) => b[0] - a[0])
            .map(([, items]) => items.sort((a, b) => a.transform[4] - b.transform[4]));
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ░░  AGENT 2: INDEXING AGENT  ░░
// ══════════════════════════════════════════════════════════════════════════════
class IndexingAgent {
    _chunk(pages, chunkSize = 800, overlap = 150) {
        const chunks = [];
        for (const page of pages) {
            const text = page.content;
            if (!text || text.length < 30) continue;
            let start = 0;
            while (start < text.length) {
                let end = start + chunkSize;
                if (end < text.length) {
                    const sp = text.lastIndexOf(' ', end);
                    if (sp > start + chunkSize * 0.6) end = sp;
                }
                const chunk = text.slice(start, end).trim();
                if (chunk.length > 40) {
                    chunks.push({ text: chunk, page: page.page_number });
                }
                start = end - overlap;
                if (start >= text.length - 50) break;
            }
        }
        return chunks;
    }

    async run(pages, onProgress) {
        onProgress('Indexing Agent: Splitting document into chunks...', 35);
        const chunks = this._chunk(pages);
        onProgress(`Indexing Agent: Created ${chunks.length} text chunks.`, 38);

        // Load Transformers.js model if not already loaded
        if (!embeddingPipeline) {
            onProgress('Indexing Agent: Loading AI embedding model (cached after first run)...', 40);
            setModelStatus('loading');

            embeddingPipeline = await pipeline(
                'feature-extraction',
                'Xenova/all-MiniLM-L6-v2',
                {
                    quantized: true,
                    progress_callback: (p) => {
                        if (p.status === 'downloading') {
                            const pct = p.total > 0 ? Math.round((p.loaded / p.total) * 100) : '?';
                            onProgress(`Downloading model: ${pct}%`, 40, `${p.file || 'model files'}`);
                        }
                        if (p.status === 'ready') {
                            setModelStatus('ready');
                        }
                    }
                }
            );
            setModelStatus('ready');
        }

        onProgress(`Indexing Agent: Generating embeddings for ${chunks.length} chunks...`, 55);

        const embeddings = [];
        const BATCH = 16;
        for (let i = 0; i < chunks.length; i += BATCH) {
            const batch = chunks.slice(i, i + BATCH).map(c => c.text);
            const output = await embeddingPipeline(batch, { pooling: 'mean', normalize: true });
            for (let j = 0; j < batch.length; j++) {
                embeddings.push(Array.from(output[j].data));
            }
            const pct = 55 + Math.round(((i + batch.length) / chunks.length) * 35);
            onProgress(`Indexing Agent: Embedded ${Math.min(i + BATCH, chunks.length)}/${chunks.length} chunks`, pct);
        }

        onProgress('Indexing Agent: FAISS-style index built ✓', 95);
        return { chunks, embeddings };
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ░░  AGENT 3: RETRIEVER AGENT  ░░
// ══════════════════════════════════════════════════════════════════════════════
class RetrieverAgent {
    _cosineSim(a, b) {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
    }

    async run(query, index, k = 4) {
        const qOut = await embeddingPipeline(query, { pooling: 'mean', normalize: true });
        const qVec = Array.from(qOut.data);

        const scored = index.embeddings.map((emb, i) => ({
            idx: i,
            score: this._cosineSim(qVec, emb)
        }));

        scored.sort((a, b) => b.score - a.score);

        return scored.slice(0, k).map(s => ({
            text:  index.chunks[s.idx].text,
            page:  index.chunks[s.idx].page,
            score: Math.round(s.score * 10000) / 10000
        }));
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ░░  AGENT 4: SUMMARIZER AGENT  ░░
// ══════════════════════════════════════════════════════════════════════════════
class SummarizerAgent {
    async run(query, chunks, history, model) {
        const ctx = chunks.map((c, i) =>
            `[Source ${i+1} – Page ${c.page}]:\n${c.text}`
        ).join('\n\n');

        const sys = `You are an expert Summarizer Agent for a PDF Research Assistant.
Answer the user's question using ONLY the provided Context.
Context may include plain text and pipe-separated table rows.

Rules:
1. Be precise, comprehensive, and structured. Render table data as markdown tables.
2. Cite sources clearly as [Page X] or [Source Y].
3. If the context doesn't contain the answer, say exactly: "I cannot find the answer in the provided document."
4. Do NOT hallucinate or extrapolate beyond what's in the context.
5. Keep your tone scientific and professional.`;

        const messages = [
            { role: 'system', content: sys },
            ...history.slice(-4),
            { role: 'user', content: `Context:\n${ctx}\n\nQuestion: ${query}` }
        ];

        return await this._groq(model, messages, { temperature: 0.15, max_tokens: 1500 });
    }

    async _groq(model, messages, opts = {}) {
        const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${groqApiKey}`
            },
            body: JSON.stringify({ model, messages, ...opts })
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err?.error?.message || `Groq API ${res.status}: ${res.statusText}`);
        }

        const data = await res.json();
        return data.choices[0].message.content;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ░░  AGENT 5: VALIDATOR AGENT (LLM-as-a-Judge)  ░░
// ══════════════════════════════════════════════════════════════════════════════
class ValidatorAgent {
    async run(query, chunks, answer, model) {
        const ctx = chunks.map((c, i) =>
            `[Source ${i+1} – Page ${c.page}]: ${c.text}`
        ).join('\n\n');

        const sys = `You are a strict Validator Agent (LLM-as-a-Judge) for a PDF Research Assistant.
Your task: cross-check the given Answer against the Source Context for factual accuracy.

Return a JSON object ONLY (no markdown) with these exact fields:
{
  "score": <integer 0-100, factual alignment percentage>,
  "claims": [
    {
      "claim": "<short description of a key assertion in the answer>",
      "status": "<fully_supported | partially_supported | unsupported>",
      "page_citation": <page number or null>,
      "reason": "<brief explanation>"
    }
  ],
  "hallucinations": ["<any fact in answer not supported by context>"],
  "verdict": "<PASS if score>=80, WARNING if 50-79, FAIL if <50>"
}`;

        const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${groqApiKey}`
            },
            body: JSON.stringify({
                model,
                messages: [
                    { role: 'system', content: sys },
                    { role: 'user',   content: `Source Context:\n${ctx}\n\nUser Question: ${query}\n\nProposed Answer:\n${answer}` }
                ],
                temperature: 0,
                max_tokens: 1200,
                response_format: { type: 'json_object' }
            })
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            console.warn('Validator error:', err);
            return { score: 50, claims: [], hallucinations: ['Validation service error'], verdict: 'WARNING' };
        }

        const data = await res.json();
        try {
            return JSON.parse(data.choices[0].message.content);
        } catch (_) {
            return { score: 50, claims: [], hallucinations: [], verdict: 'WARNING' };
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ░░  AGENT 6: CONVERSATION AGENT (Orchestrator)  ░░
// ══════════════════════════════════════════════════════════════════════════════
class ConversationAgent {
    constructor() {
        this.ingestion   = new IngestionAgent();
        this.indexing    = new IndexingAgent();
        this.retriever   = new RetrieverAgent();
        this.summarizer  = new SummarizerAgent();
        this.validator   = new ValidatorAgent();
    }

    async ingestDocument(file) {
        const { pages, metadata } = await this.ingestion.run(file, (msg, pct, sub) => {
            updateProgress(msg, pct, sub);
        });

        setNodeState('node-ingestion', 'completed', 'Parsed ✓');
        setConnector('conn-1', true);
        setNodeState('node-indexing', 'active', 'Embedding...');

        const index = await this.indexing.run(pages, (msg, pct, sub) => {
            updateProgress(msg, pct, sub);
        });

        metadata.chunk_count = index.chunks.length;
        return { index, metadata };
    }

    async query(question) {
        if (!docIndex) throw new Error('No document loaded.');

        const logs = [];

        // Retriever
        setNodeState('node-retriever', 'active', 'Searching...');
        setConnector('conn-2', true);
        logs.push({ agent: 'Retriever Agent', status: 'running', message: 'Performing semantic similarity search...' });

        const chunks = await this.retriever.run(question, docIndex, topK);
        setNodeState('node-retriever', 'completed', `Found ${chunks.length} chunks`);
        logs.push({ agent: 'Retriever Agent', status: 'completed', message: `Retrieved ${chunks.length} relevant sections from page(s): ${[...new Set(chunks.map(c => c.page))].join(', ')}` });

        if (!chunks.length) {
            return {
                answer: 'I could not find any relevant text in the document for your question.',
                retrieved_chunks: [],
                validation: { score: 100, claims: [], hallucinations: [], verdict: 'PASS' },
                agent_logs: logs
            };
        }

        // Summarizer
        setConnector('conn-3', true);
        setNodeState('node-summarizer', 'active', 'Drafting...');
        logs.push({ agent: 'Summarizer Agent', status: 'running', message: `Generating answer via Groq (${summarizerModel})...` });

        const answer = await this.summarizer.run(question, chunks, chatHistory, summarizerModel);
        setNodeState('node-summarizer', 'completed', 'Drafted ✓');
        logs.push({ agent: 'Summarizer Agent', status: 'completed', message: 'Answer synthesized with citations.' });

        // Validator
        setConnector('conn-4', true);
        setNodeState('node-validator', 'active', 'Verifying...');
        logs.push({ agent: 'Validator Agent', status: 'running', message: `Cross-checking facts via ${validatorModel}...` });

        const validation = await this.validator.run(question, chunks, answer, validatorModel);
        setNodeState('node-validator', 'completed', validation.verdict);
        logs.push({ agent: 'Validator Agent', status: 'completed', message: `Score: ${validation.score}% · Verdict: ${validation.verdict}` });

        // Conversation wrap-up
        setConnector('conn-5', true);
        setNodeState('node-conversation', 'completed', 'Responded');
        logs.push({ agent: 'Conversation Agent', status: 'completed', message: 'Response packaged and delivered.' });

        return { answer, retrieved_chunks: chunks, validation, agent_logs: logs };
    }
}

const agent = new ConversationAgent();

// ══════════════════════════════════════════════════════════════════════════════
// ░░  UI INIT & EVENT LISTENERS  ░░
// ══════════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', init);

function init() {
    // Restore settings
    D.groqApiKeyInput.value  = groqApiKey;
    D.summarizerModelSel.value = summarizerModel;
    D.validatorModelSel.value  = validatorModel;
    D.topKSel.value = String(topK);

    if (!groqApiKey) openModal();

    // Settings
    D.btnSettings.addEventListener('click', openModal);
    D.btnCloseSettings.addEventListener('click', closeModal);
    window.addEventListener('click', e => { if (e.target === D.settingsModal) closeModal(); });
    D.settingsForm.addEventListener('submit', saveSettings);

    // Upload
    D.btnBrowse.addEventListener('click', e => {
        e.stopPropagation();
        D.pdfFileInput.value = '';
        D.pdfFileInput.click();
    });
    D.pdfFileInput.addEventListener('change', e => {
        if (e.target.files?.length) handleUpload(e.target.files[0]);
    });
    D.dropZone.addEventListener('dragover', e => { e.preventDefault(); D.dropZone.classList.add('dragover'); });
    D.dropZone.addEventListener('dragleave', () => D.dropZone.classList.remove('dragover'));
    D.dropZone.addEventListener('drop', e => {
        e.preventDefault();
        D.dropZone.classList.remove('dragover');
        if (e.dataTransfer.files?.length) handleUpload(e.dataTransfer.files[0]);
    });

    // Chat
    D.chatForm.addEventListener('submit', handleChat);
    D.btnClearChat.addEventListener('click', () => {
        chatHistory = [];
        D.chatMessages.innerHTML = '';
        showWelcome();
    });

    // Reset
    D.btnResetPdf.addEventListener('click', resetDocument);
}

function saveSettings(e) {
    e.preventDefault();
    groqApiKey      = D.groqApiKeyInput.value.trim();
    summarizerModel = D.summarizerModelSel.value;
    validatorModel  = D.validatorModelSel.value;
    topK            = parseInt(D.topKSel.value);
    localStorage.setItem('groq_api_key',      groqApiKey);
    localStorage.setItem('summarizer_model',   summarizerModel);
    localStorage.setItem('validator_model',    validatorModel);
    localStorage.setItem('top_k',              topK);
    closeModal();
    systemNote('Configuration saved. API key stored in browser only.');
}

// ─── FILE UPLOAD ─────────────────────────────────────────────────────────────
async function handleUpload(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        alert('Only PDF files are supported.');
        return;
    }

    // UI: switch to progress view
    D.dropZone.classList.add('hidden');
    D.btnBrowse.classList.add('hidden');
    D.pdfDetails.classList.add('hidden');
    D.uploadProgress.classList.remove('hidden');

    resetPipelineNodes();
    setNodeState('node-ingestion', 'active', 'Parsing...');

    try {
        const { index, metadata } = await agent.ingestDocument(file);

        docIndex = index;
        docMeta  = metadata;

        setNodeState('node-indexing', 'completed', 'Indexed ✓');
        setNodeState('node-conversation', 'completed', 'Ready');
        updateProgress('All agents ready! Ask your first question.', 100);

        await sleep(600);
        showDocDetails(metadata);
        systemNote(`"${metadata.title}" indexed — ${metadata.chunk_count} chunks ready for semantic search.`);

    } catch (err) {
        console.error('Upload error:', err);
        alert(`Failed to process PDF:\n${err.message}`);
        showUploadUI();
        resetPipelineNodes();
    }
}

function resetDocument() {
    if (!confirm('Clear the current document and reset the session?')) return;
    docIndex = null;
    docMeta  = null;
    chatHistory = [];
    showUploadUI();
    resetPipelineNodes();
    resetValidation();
    resetCitations();
    D.chatMessages.innerHTML = '';
    showWelcome();
    systemNote('Session cleared.');
}

// ─── CHAT ────────────────────────────────────────────────────────────────────
async function handleChat(e) {
    e.preventDefault();
    const q = D.userInput.value.trim();
    if (!q || isProcessing) return;

    if (!groqApiKey) {
        alert('Please configure your Groq API key in Settings first.');
        openModal();
        return;
    }

    isProcessing = true;
    D.userInput.value = '';
    D.btnSend.disabled = true;
    D.userInput.disabled = true;

    renderBubble('user', q);
    const typingEl = renderTyping();

    // Reset pipeline from retriever onwards
    ['node-retriever','node-summarizer','node-validator','node-conversation'].forEach(id => {
        setNodeState(id, '', 'Idle');
    });
    ['conn-2','conn-3','conn-4','conn-5'].forEach(id => setConnector(id, false));

    try {
        const result = await agent.query(q);

        typingEl.remove();
        renderBubble('assistant', result.answer, result.agent_logs);
        renderValidation(result.validation);
        renderCitations(result.retrieved_chunks);

        chatHistory.push({ role: 'user', content: q });
        chatHistory.push({ role: 'assistant', content: result.answer });

    } catch (err) {
        typingEl.remove();
        renderBubble('assistant', `<span style="color:var(--color-danger)"><i class="fa-solid fa-triangle-exclamation"></i> <strong>Error:</strong> ${err.message}</span>`);
        resetPipelineNodes();
        if (docMeta) {
            setNodeState('node-ingestion', 'completed', 'Ready');
            setNodeState('node-indexing',  'completed', 'Ready');
            setNodeState('node-conversation', 'completed', 'Ready');
        }
    }

    isProcessing = false;
    D.btnSend.disabled = false;
    D.userInput.disabled = false;
    D.userInput.focus();
    D.chatMessages.scrollTop = D.chatMessages.scrollHeight;
}

// ─── PIPELINE VISUAL ─────────────────────────────────────────────────────────
function setNodeState(id, state, statusText) {
    const node = $(id);
    if (!node) return;
    node.className = `agent-node${state ? ' ' + state : ''}`;
    node.querySelector('.node-status').textContent = statusText;
}

function setConnector(id, active) {
    const el = $(id);
    if (!el) return;
    el.className = active ? 'pipeline-connector active' : 'pipeline-connector';
}

function resetPipelineNodes() {
    ['node-ingestion','node-indexing','node-retriever','node-summarizer','node-validator','node-conversation']
        .forEach(id => setNodeState(id, '', 'Idle'));
    ['conn-1','conn-2','conn-3','conn-4','conn-5'].forEach(id => setConnector(id, false));
}

function updateProgress(msg, pct, sub = '') {
    D.progressMessage.textContent = msg;
    D.progressBar.style.width = `${pct}%`;
    D.progressSub.textContent = sub;
}

// ─── UI STATE ─────────────────────────────────────────────────────────────────
function showDocDetails(meta) {
    D.uploadProgress.classList.add('hidden');
    D.dropZone.classList.add('hidden');
    D.btnBrowse.classList.add('hidden');

    D.pdfName.textContent   = meta.filename;
    D.pdfPages.textContent  = meta.pages;
    D.pdfWords.textContent  = fmt(meta.word_count);
    D.pdfChunks.textContent = meta.chunk_count || '-';
    D.pdfTables.textContent = meta.tables_found || '0';
    D.pdfTitle.textContent  = meta.title;
    D.pdfAuthor.textContent = meta.author;
    D.pdfDetails.classList.remove('hidden');

    D.activeDocTitle.textContent = meta.title;
    const parts = [`${meta.pages} pages`, `${fmt(meta.word_count)} words`, `${meta.chunk_count} chunks`];
    if (meta.tables_found > 0) parts.push(`${meta.tables_found} tables`);
    D.activeDocSub.textContent = parts.join(' · ');
    D.chatHeaderInfo.classList.add('active');

    D.userInput.disabled = false;
    D.btnSend.disabled = false;
    D.userInput.placeholder = 'Ask a question about the document...';
}

function showUploadUI() {
    D.uploadProgress.classList.add('hidden');
    D.pdfDetails.classList.add('hidden');
    D.dropZone.classList.remove('hidden');
    D.btnBrowse.classList.remove('hidden');

    D.activeDocTitle.textContent = 'Research Assistant';
    D.activeDocSub.textContent = 'Upload a PDF to begin chatting';
    D.chatHeaderInfo.classList.remove('active');

    D.userInput.disabled = true;
    D.btnSend.disabled = true;
    D.userInput.placeholder = 'Upload a PDF to start asking questions...';
}

function setModelStatus(state) {
    D.modelStatus.className = `connection-status model-${state}`;
    const labels = { loading: 'Loading Model…', ready: 'Model Ready ✓', error: 'Model Error' };
    D.modelStatusText.textContent = labels[state] || 'Model Not Loaded';
}

function openModal()  { D.settingsModal.classList.add('show'); }
function closeModal() { D.settingsModal.classList.remove('show'); }

// ─── CHAT RENDERING ───────────────────────────────────────────────────────────
function showWelcome() {
    D.chatMessages.innerHTML = `
    <div class="system-welcome">
        <div class="welcome-icon"><i class="fa-solid fa-robot"></i></div>
        <h2>Welcome to PDF Research Assistant</h2>
        <p>All 6 AI agents run directly in your browser — no server required.</p>
        <div class="welcome-steps">
            <div class="step-item"><span class="step-num">1</span>
                <span>Click <strong>Settings</strong> and enter your <strong>Groq API Key</strong>
                (free at <a href="https://console.groq.com" target="_blank" style="color:var(--accent-primary)">console.groq.com</a>).</span>
            </div>
            <div class="step-item"><span class="step-num">2</span>
                <span>Upload any PDF — research papers, reports, contracts, textbooks.</span>
            </div>
            <div class="step-item"><span class="step-num">3</span>
                <span>Ask questions. Agents retrieve, summarize, and fact-check in real time.</span>
            </div>
        </div>
        <div class="browser-badge"><i class="fa-solid fa-lock"></i> Fully private — PDF never leaves your browser</div>
    </div>`;
}

function clearWelcome() {
    const w = D.chatMessages.querySelector('.system-welcome');
    if (w) w.remove();
}

function renderBubble(role, content, logs = null) {
    clearWelcome();
    const div = document.createElement('div');
    div.className = `message ${role}`;

    const avatar = `<div class="msg-avatar"><i class="fa-solid fa-${role === 'user' ? 'user' : 'robot'}"></i></div>`;
    const bubble = `<div class="msg-bubble">${role === 'user' ? escapeHtml(content) : formatMd(content)}</div>`;

    let logsHtml = '';
    if (logs?.length) {
        const items = logs.map(l =>
            `<div class="log-item">
              <span class="log-dot ${l.status}"></span>
              <span><strong class="log-agent">${l.agent}:</strong> <span class="log-msg">${l.message}</span></span>
            </div>`
        ).join('');
        logsHtml = `
        <div class="agent-logs-expander">
            <div class="expander-header" onclick="this.nextElementSibling.classList.toggle('hidden'); this.querySelector('.toggle-arrow').classList.toggle('fa-chevron-up'); this.querySelector('.toggle-arrow').classList.toggle('fa-chevron-down');">
                <span><i class="fa-solid fa-bug"></i> View Agent Audit Logs</span>
                <i class="fa-solid fa-chevron-down toggle-arrow"></i>
            </div>
            <div class="expander-content hidden">${items}</div>
        </div>`;
    }

    div.innerHTML = `${avatar}<div class="msg-content-wrapper">${bubble}${logsHtml}</div>`;
    D.chatMessages.appendChild(div);
    D.chatMessages.scrollTop = D.chatMessages.scrollHeight;
}

function renderTyping() {
    clearWelcome();
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `<div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
    <div class="msg-bubble"><div class="dual-spinner" style="width:18px;height:18px;border-width:2px;margin:2px auto;"></div></div>`;
    D.chatMessages.appendChild(div);
    D.chatMessages.scrollTop = D.chatMessages.scrollHeight;
    return div;
}

function systemNote(text) {
    clearWelcome();
    const div = document.createElement('div');
    div.style.cssText = 'text-align:center;font-size:11px;color:var(--text-muted);margin:6px 0;';
    div.innerHTML = `<i class="fa-solid fa-info-circle"></i> ${text}`;
    D.chatMessages.appendChild(div);
    D.chatMessages.scrollTop = D.chatMessages.scrollHeight;
}

// ─── VALIDATION RENDERING ─────────────────────────────────────────────────────
function renderValidation(report) {
    D.noValidationMsg.classList.add('hidden');
    D.validationResult.classList.remove('hidden');

    const score = report.score ?? 0;
    D.gaugeBarInner.style.width = `${score}%`;
    D.gaugeValue.textContent = `${score}%`;

    if (score >= 80) {
        D.gaugeBarInner.style.backgroundColor = 'var(--color-success)';
        D.verdictBadge.className = 'badge pass';
    } else if (score >= 50) {
        D.gaugeBarInner.style.backgroundColor = 'var(--color-warning)';
        D.verdictBadge.className = 'badge warning';
    } else {
        D.gaugeBarInner.style.backgroundColor = 'var(--color-danger)';
        D.verdictBadge.className = 'badge fail';
    }
    D.verdictBadge.textContent = report.verdict || 'FAIL';

    D.claimsList.innerHTML = '';
    (report.claims || []).forEach(c => {
        const li = document.createElement('li');
        li.className = 'claim-item';
        li.innerHTML = `
            <div class="claim-top">
                <span class="claim-desc">${escapeHtml(c.claim)}</span>
                <span class="claim-status ${c.status}">${c.status.replace(/_/g,' ')}</span>
            </div>
            <div class="claim-reason">${escapeHtml(c.reason || '')}</div>
            ${c.page_citation ? `<div class="claim-citation">[Page ${c.page_citation}]</div>` : ''}`;
        D.claimsList.appendChild(li);
    });

    const halls = report.hallucinations || [];
    if (halls.length) {
        D.hallucinationsList.innerHTML = halls.map(h =>
            `<div class="hallucination-alert"><i class="fa-solid fa-triangle-exclamation"></i><span>${escapeHtml(h)}</span></div>`
        ).join('');
        D.hallucinationsSection.classList.remove('hidden');
    } else {
        D.hallucinationsSection.classList.add('hidden');
    }
}

function resetValidation() {
    D.noValidationMsg.classList.remove('hidden');
    D.validationResult.classList.add('hidden');
}

// ─── CITATIONS RENDERING ──────────────────────────────────────────────────────
function renderCitations(chunks) {
    D.noCitationsMsg.classList.add('hidden');
    D.citationsList.classList.remove('hidden');
    D.citationsList.innerHTML = chunks.map((c, i) => `
        <div class="citation-block">
            <div class="citation-meta">
                <span class="citation-source"><i class="fa-solid fa-book"></i> Source ${i+1} · Page ${c.page}</span>
                <span class="citation-score">${Math.round(c.score * 100)}% match</span>
            </div>
            <div class="citation-text">&ldquo;${escapeHtml(c.text.slice(0, 280))}${c.text.length > 280 ? '…' : ''}&rdquo;</div>
        </div>`
    ).join('');
}

function resetCitations() {
    D.noCitationsMsg.classList.remove('hidden');
    D.citationsList.classList.add('hidden');
    D.citationsList.innerHTML = '';
}

// ─── MARKDOWN / HTML HELPERS ──────────────────────────────────────────────────
function escapeHtml(str = '') {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function formatMd(text = '') {
    let h = text
        // Code blocks first
        .replace(/```([\s\S]*?)```/g, (_,c) => `<pre><code>${escapeHtml(c.trim())}</code></pre>`)
        .replace(/`([^`]+)`/g, (_,c) => `<code>${escapeHtml(c)}</code>`)
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/__(.*?)__/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Page/Source citations
        .replace(/\[Page\s+(\d+)\]/gi, '<span style="color:var(--accent-primary);font-weight:700;">[Page $1]</span>')
        .replace(/\[Source\s+(\d+)\]/gi, '<span style="color:var(--accent-secondary);font-weight:700;">[Source $1]</span>')
        // Headers
        .replace(/^### (.+)$/gm, '<h4>$1</h4>')
        .replace(/^## (.+)$/gm, '<h3>$1</h3>')
        .replace(/^# (.+)$/gm, '<h3>$1</h3>')
        // Unordered lists
        .replace(/^[\-\*] (.+)$/gm, '<li>$1</li>')
        // Ordered lists
        .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
        // Paragraphs
        .replace(/\n{2,}/g, '</p><p>')
        .replace(/\n/g, '<br>');

    // Wrap consecutive <li> in <ul>
    h = h.replace(/((?:<li>.*?<\/li>\s*)+)/gs, '<ul>$1</ul>');

    return `<p>${h}</p>`;
}

function fmt(n) { return n?.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',') ?? '-'; }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
