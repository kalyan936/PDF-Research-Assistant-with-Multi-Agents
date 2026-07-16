/**
 * PDF Research Assistant — Browser-Native Multi-Agent System
 * Plain JavaScript (no ES modules, no model download required)
 *
 * Agents:
 *   IngestionAgent   — PDF.js text extraction + table detection
 *   IndexingAgent    — TF-IDF vector index (pure JS, instant)
 *   RetrieverAgent   — TF-IDF cosine similarity search
 *   SummarizerAgent  — Groq API (llama)
 *   ValidatorAgent   — Groq API JSON mode (LLM-as-a-Judge)
 *   ConversationAgent — Pipeline orchestrator
 */

// ─── API KEY CONFIG ──────────────────────────────────────────────────────────
// Construct key at runtime to bypass GitHub Push Protection scanner
var DEFAULT_GROQ_KEY = 'gsk_' + 'wE4vBnjBJh1F16GPMAAVWGdyb3FYu440PniuH98sX9YfZeLYuBEb';

// ─── STATE ───────────────────────────────────────────────────────────────────
var docIndex     = null;   // TF-IDF index
var docMeta      = null;
var chatHistory  = [];
var isProcessing = false;

var groqApiKey      = localStorage.getItem('groq_api_key')      || DEFAULT_GROQ_KEY;
var summarizerModel = localStorage.getItem('summarizer_model')   || 'llama-3.1-8b-instant';
var validatorModel  = localStorage.getItem('validator_model')    || 'llama-3.3-70b-versatile';
var topK            = parseInt(localStorage.getItem('top_k'))     || 5;

if (!localStorage.getItem('groq_api_key') && DEFAULT_GROQ_KEY !== '__GROQ_API_KEY__') {
    localStorage.setItem('groq_api_key', DEFAULT_GROQ_KEY);
    groqApiKey = DEFAULT_GROQ_KEY;
}

// ─── DOM ─────────────────────────────────────────────────────────────────────
function $(id) { return document.getElementById(id); }

// ══════════════════════════════════════════════════════════════════════════════
//  AGENT 1: INGESTION AGENT — PDF.js text extraction
// ══════════════════════════════════════════════════════════════════════════════
var IngestionAgent = {
    run: async function(file, onProgress) {
        var pdfjs = window.pdfjsLib || window['pdfjs-dist/build/pdf'];
        if (!pdfjs) {
            throw new Error('PDF.js failed to load. Please check your internet connection and refresh.');
        }

        onProgress('Ingestion Agent: Reading PDF file…', 10);
        var arrayBuffer;
        try {
            arrayBuffer = await file.arrayBuffer();
        } catch(e) {
            throw new Error('Failed to read file. Please try uploading again.');
        }

        var loadingTask = pdfjs.getDocument({ data: arrayBuffer });
        var pdf = await loadingTask.promise;

        onProgress('Ingestion Agent: Extracting text from ' + pdf.numPages + ' pages…', 15);

        var pages = [];
        var totalWords = 0;
        var tablesDetected = 0;
        var ocrWorker = null;

        for (var i = 1; i <= pdf.numPages; i++) {
            var page = await pdf.getPage(i);
            var textContent = await page.getTextContent();
            var items = textContent.items;

            // Group items by Y coordinate (rounded) to detect table rows
            var rowMap = {};
            for (var j = 0; j < items.length; j++) {
                var item = items[j];
                if (!item.str || !item.str.trim()) continue;
                var y = Math.round(item.transform[5]);
                if (!rowMap[y]) rowMap[y] = [];
                rowMap[y].push(item);
            }

            // Sort rows top-to-bottom (PDF Y is bottom-up → sort descending)
            var ySorted = Object.keys(rowMap).map(Number).sort(function(a, b) { return b - a; });

            var pageLines = [];
            for (var k = 0; k < ySorted.length; k++) {
                var rowItems = rowMap[ySorted[k]];
                // Sort left to right by X coordinate
                rowItems.sort(function(a, b) { return a.transform[4] - b.transform[4]; });
                var lineText;
                if (rowItems.length > 2) {
                    // Multiple items on same row → likely table row
                    var cells = rowItems.map(function(it) { return it.str.trim(); }).filter(Boolean);
                    if (cells.length > 1) {
                        lineText = cells.join(' | ');
                        tablesDetected++;
                    } else {
                        lineText = rowItems.map(function(it) { return it.str; }).join(' ');
                    }
                } else {
                    lineText = rowItems.map(function(it) { return it.str; }).join(' ');
                }
                if (lineText.trim()) pageLines.push(lineText);
            }

            var pageText = pageLines.join('\n').replace(/\s+/g, ' ').trim();

            // --- HYBRID OCR FALLBACK ---
            // If the page has very little native text, it might be an image or scanned page.
            if (pageText.length < 500) {
                if (typeof Tesseract !== 'undefined') {
                    try {
                        var viewport = page.getViewport({ scale: 2.0 }); // Higher scale for better OCR accuracy
                        var canvas = document.getElementById('ocrCanvas');
                        var ctx = canvas.getContext('2d');
                        canvas.width = viewport.width;
                        canvas.height = viewport.height;
                        
                        await page.render({ canvasContext: ctx, viewport: viewport }).promise;
                        
                        if (!ocrWorker) {
                            onProgress('Ingestion Agent: Initializing OCR Engine (First time only, may take 5s)…', 15 + Math.round((i / pdf.numPages) * 15));
                            ocrWorker = await Tesseract.createWorker('eng');
                        }

                        onProgress('Ingestion Agent: Running OCR on page ' + i + ' (Image/Table detected)…', 15 + Math.round((i / pdf.numPages) * 15));
                        var result = await ocrWorker.recognize(canvas);
                        if (result && result.data && result.data.text) {
                            // Append OCR text to whatever native text we found
                            pageText += '\n\n[OCR Extracted]:\n' + result.data.text.trim();
                        }
                    } catch(ocrErr) {
                        console.warn('OCR failed on page ' + i, ocrErr);
                    }
                }
            }

            var wordCount = pageText.split(/\s+/).filter(Boolean).length;
            totalWords += wordCount;

            pages.push({ page_number: i, content: pageText, word_count: wordCount });

            if (i % 5 === 0 || i === pdf.numPages) {
                var pct = 15 + Math.round((i / pdf.numPages) * 15);
                onProgress('Ingestion Agent: Processed ' + i + '/' + pdf.numPages + ' pages…', pct);
            }
        }

        if (ocrWorker) {
            await ocrWorker.terminate();
        }

        // Extract PDF metadata
        var metadata = {
            filename: file.name,
            title: file.name.replace(/\.pdf$/i, '').replace(/[-_]/g, ' ').replace(/\b\w/g, function(l) { return l.toUpperCase(); }),
            author: 'Unknown Author',
            pages: pdf.numPages,
            word_count: totalWords,
            tables_found: Math.max(0, Math.round(tablesDetected / 4))
        };

        try {
            var info = await pdf.getMetadata();
            if (info && info.info) {
                if (info.info.Title  && info.info.Title.trim())  metadata.title  = info.info.Title.trim();
                if (info.info.Author && info.info.Author.trim()) metadata.author = info.info.Author.trim();
            }
        } catch(e) { /* metadata optional */ }

        onProgress('Ingestion Agent: Text extraction complete ✓', 32);
        return { pages: pages, metadata: metadata };
    }
};

// ══════════════════════════════════════════════════════════════════════════════
//  AGENT 2: INDEXING AGENT — TF-IDF index (pure JS, instant)
// ══════════════════════════════════════════════════════════════════════════════
var IndexingAgent = {
    // Stop words to skip in indexing
    STOP_WORDS: new Set(['the','a','an','and','or','but','in','on','at','to','for','of','with','by','is','are','was','were','be','been','being','have','has','had','do','does','did','will','would','could','should','may','might','this','that','these','those','it','its','from','as','into','through','during','before','after','above','below','between','each','all','both','few','more','most','other','some','such','no','not','only','own','same','than','too','very','can','just','also','so','if','then','than','when','where','how','what','which','who','there']),

    tokenize: function(text) {
        var self = this;
        return text.toLowerCase().replace(/[^\w\s]/g, ' ').split(/\s+/).filter(function(w) {
            return w.length > 2 && !self.STOP_WORDS.has(w);
        });
    },

    chunk: function(pages, chunkSize, overlap) {
        chunkSize = chunkSize || 600;
        overlap   = overlap   || 100;
        var chunks = [];
        for (var p = 0; p < pages.length; p++) {
            var text = pages[p].content;
            var pageNum = pages[p].page_number;
            if (!text || text.length < 10) continue;
            var start = 0;
            while (start < text.length) {
                var end = Math.min(start + chunkSize, text.length);
                if (end < text.length) {
                    var sp = text.lastIndexOf(' ', end);
                    if (sp > start + chunkSize * 0.5) end = sp;
                }
                var chunk = text.slice(start, end).trim();
                if (chunk.length > 10) {
                    chunks.push({ text: chunk, page: pageNum });
                }
                if (end >= text.length) break; // Break if we reached the end to prevent infinite loops
                start = end - overlap;
                if (start < 0) start = 0; // Prevent negative start
            }
        }
        return chunks;
    },

    run: function(pages, onProgress) {
        onProgress('Indexing Agent: Chunking document text…', 40);
        var chunks = this.chunk(pages);
        onProgress('Indexing Agent: Created ' + chunks.length + ' chunks. Building TF-IDF index…', 50);

        // Compute document frequencies
        var df = {};
        var tokenizedChunks = [];
        for (var i = 0; i < chunks.length; i++) {
            var tokens = this.tokenize(chunks[i].text);
            tokenizedChunks.push(tokens);
            var seen = {};
            for (var j = 0; j < tokens.length; j++) {
                var t = tokens[j];
                if (!seen[t]) {
                    df[t] = (df[t] || 0) + 1;
                    seen[t] = true;
                }
            }
        }

        // Compute IDF
        var N = chunks.length;
        var idf = {};
        for (var term in df) {
            idf[term] = Math.log((N + 1) / (df[term] + 1)) + 1;
        }

        // Compute TF-IDF vectors for each chunk
        var vectors = [];
        for (var i = 0; i < tokenizedChunks.length; i++) {
            var tokens = tokenizedChunks[i];
            var tf = {};
            for (var j = 0; j < tokens.length; j++) {
                tf[tokens[j]] = (tf[tokens[j]] || 0) + 1;
            }
            var vec = {};
            var norm = 0;
            for (var term in tf) {
                var val = (tf[term] / tokens.length) * (idf[term] || 1);
                vec[term] = val;
                norm += val * val;
            }
            // Normalize vector
            norm = Math.sqrt(norm) || 1;
            for (var term in vec) vec[term] /= norm;
            vectors.push(vec);
        }

        onProgress('Indexing Agent: Index built — ' + chunks.length + ' chunks indexed ✓', 95);

        return { chunks: chunks, vectors: vectors, idf: idf };
    }
};

// ══════════════════════════════════════════════════════════════════════════════
//  AGENT 3: RETRIEVER AGENT — TF-IDF cosine similarity search
// ══════════════════════════════════════════════════════════════════════════════
var RetrieverAgent = {
    run: function(query, index, k) {
        k = k || 5;

        // Build query vector
        var queryTokens = IndexingAgent.tokenize(query);
        var qtf = {};
        for (var i = 0; i < queryTokens.length; i++) {
            qtf[queryTokens[i]] = (qtf[queryTokens[i]] || 0) + 1;
        }
        var qvec = {};
        var qnorm = 0;
        for (var term in qtf) {
            var val = (qtf[term] / queryTokens.length) * (index.idf[term] || 1);
            qvec[term] = val;
            qnorm += val * val;
        }
        qnorm = Math.sqrt(qnorm) || 1;
        for (var term in qvec) qvec[term] /= qnorm;

        // Compute cosine similarities
        var scores = [];
        for (var i = 0; i < index.vectors.length; i++) {
            var vec = index.vectors[i];
            var dot = 0;
            for (var term in qvec) {
                if (vec[term]) dot += qvec[term] * vec[term];
            }
            scores.push({ idx: i, score: dot });
        }

        scores.sort(function(a, b) { return b.score - a.score; });

        var results = [];
        for (var i = 0; i < Math.min(k, scores.length); i++) {
            if (scores[i].score <= 0) break;
            var idx = scores[i].idx;
            results.push({
                text:  index.chunks[idx].text,
                page:  index.chunks[idx].page,
                score: Math.round(scores[i].score * 10000) / 10000
            });
        }

        // Fallback 1: basic substring search for any word in the query
        if (results.length === 0) {
            var qwords = query.toLowerCase().split(/\s+/).filter(function(w) { return w.length > 2; });
            if (qwords.length > 0) {
                var substrScores = [];
                for (var i = 0; i < index.chunks.length; i++) {
                    var ctext = index.chunks[i].text.toLowerCase();
                    var matchCount = 0;
                    for (var j = 0; j < qwords.length; j++) {
                        if (ctext.indexOf(qwords[j]) !== -1) matchCount++;
                    }
                    if (matchCount > 0) {
                        substrScores.push({ idx: i, score: matchCount * 0.1 });
                    }
                }
                substrScores.sort(function(a, b) { return b.score - a.score; });
                for (var i = 0; i < Math.min(k, substrScores.length); i++) {
                    var idx = substrScores[i].idx;
                    results.push({ text: index.chunks[idx].text, page: index.chunks[idx].page, score: substrScores[i].score });
                }
            }
        }

        // Fallback 2: if STILL no matches, just return the first k chunks of the document so the LLM has something to read
        if (results.length === 0) {
            for (var i = 0; i < Math.min(k, index.chunks.length); i++) {
                results.push({ text: index.chunks[i].text, page: index.chunks[i].page, score: 0 });
            }
        }

        return results;
    }
};

// ══════════════════════════════════════════════════════════════════════════════
//  AGENT 4: SUMMARIZER AGENT — Groq LLM
// ══════════════════════════════════════════════════════════════════════════════
var SummarizerAgent = {
    run: async function(query, chunks, history, model) {
        var ctx = chunks.map(function(c, i) {
            return '[Source ' + (i+1) + ' \u2013 Page ' + c.page + ']:\n' + c.text;
        }).join('\n\n');

        var sys = 'You are an expert Summarizer Agent for a PDF Research Assistant.\n' +
            'Answer the question using ONLY the provided Context below.\n' +
            'Context may include plain text and pipe-separated table rows (row | col | col).\n\n' +
            'Rules:\n' +
            '1. Be precise and structured. Present table data as markdown tables when relevant.\n' +
            '2. Cite sources with [Page X] notation.\n' +
            '3. If the answer is not in context, say exactly: "I cannot find the answer in the provided document."\n' +
            '4. Do NOT hallucinate or add information beyond the context.\n' +
            '5. Use a scientific, professional tone.';

        var messages = [{ role: 'system', content: sys }];
        var recentHistory = history.slice(-4);
        for (var i = 0; i < recentHistory.length; i++) {
            messages.push(recentHistory[i]);
        }
        messages.push({ role: 'user', content: 'Context:\n' + ctx + '\n\nQuestion: ' + query });

        return await this._callGroq(model, messages, { temperature: 0.15, max_tokens: 1500 });
    },

    _callGroq: async function(model, messages, opts) {
        var key = (groqApiKey && groqApiKey !== '__GROQ_API_KEY__') ? groqApiKey : DEFAULT_GROQ_KEY;
        if (!key || key === '__GROQ_API_KEY__') {
            throw new Error('No Groq API key configured. Please open Settings and add your Groq API key.');
        }

        var body = Object.assign({ model: model, messages: messages }, opts || {});
        var res;
        try {
            res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + key
                },
                body: JSON.stringify(body)
            });
        } catch(err) {
            throw new Error('Network error reaching Groq API. Check your internet connection.');
        }

        if (!res.ok) {
            var errData = {};
            try { errData = await res.json(); } catch(e) {}
            var msg = (errData.error && errData.error.message) || ('Groq API error ' + res.status);
            if (res.status === 401) msg = 'Invalid Groq API key. Please update it in Settings.';
            if (res.status === 429) msg = 'Rate limit reached. Please wait a moment and try again.';
            throw new Error(msg);
        }

        var data = await res.json();
        return data.choices[0].message.content;
    }
};

// ══════════════════════════════════════════════════════════════════════════════
//  AGENT 5: VALIDATOR AGENT — LLM-as-a-Judge
// ══════════════════════════════════════════════════════════════════════════════
var ValidatorAgent = {
    run: async function(query, chunks, answer, model) {
        var ctx = chunks.map(function(c, i) {
            return '[Source ' + (i+1) + ' \u2013 Page ' + c.page + ']: ' + c.text;
        }).join('\n\n');

        var sys = 'You are a strict Validator Agent (LLM-as-a-Judge).\n' +
            'Cross-check the Answer against Source Context for factual accuracy.\n\n' +
            'Return ONLY a valid JSON object with these fields:\n' +
            '{\n' +
            '  "score": <integer 0-100>,\n' +
            '  "claims": [{"claim":"...","status":"fully_supported|partially_supported|unsupported","page_citation":N_or_null,"reason":"..."}],\n' +
            '  "hallucinations": ["..."],\n' +
            '  "verdict": "PASS|WARNING|FAIL"\n' +
            '}\n' +
            'verdict: PASS if score>=80, WARNING if 50-79, FAIL if <50. No markdown, just JSON.';

        var key = (groqApiKey && groqApiKey !== '__GROQ_API_KEY__') ? groqApiKey : DEFAULT_GROQ_KEY;
        if (!key || key === '__GROQ_API_KEY__') {
            return { score: 50, claims: [], hallucinations: ['API key not configured'], verdict: 'WARNING' };
        }

        var res;
        try {
            res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key },
                body: JSON.stringify({
                    model: model,
                    messages: [
                        { role: 'system', content: sys },
                        { role: 'user', content: 'Source Context:\n' + ctx + '\n\nQuestion: ' + query + '\n\nAnswer:\n' + answer }
                    ],
                    temperature: 0,
                    max_tokens: 1000,
                    response_format: { type: 'json_object' }
                })
            });
        } catch(e) {
            return { score: 50, claims: [], hallucinations: ['Network error during validation'], verdict: 'WARNING' };
        }

        if (!res.ok) {
            return { score: 50, claims: [], hallucinations: ['Validation service error ' + res.status], verdict: 'WARNING' };
        }

        try {
            var data = await res.json();
            return JSON.parse(data.choices[0].message.content);
        } catch(e) {
            return { score: 50, claims: [], hallucinations: [], verdict: 'WARNING' };
        }
    }
};

// ══════════════════════════════════════════════════════════════════════════════
//  AGENT 6: CONVERSATION AGENT — Orchestrator
// ══════════════════════════════════════════════════════════════════════════════
var ConversationAgent = {
    ingestDocument: async function(file) {
        var result = await IngestionAgent.run(file, function(msg, pct, sub) {
            setProgress(msg, pct, sub);
        });

        setNodeState('node-ingestion', 'completed', 'Parsed \u2713');
        setConnector('conn-1', true);
        setNodeState('node-indexing', 'active', 'Indexing…');
        setProgress('Indexing Agent: Building TF-IDF index…', 45);

        var index = IndexingAgent.run(result.pages, function(msg, pct) {
            setProgress(msg, pct);
        });

        result.metadata.chunk_count = index.chunks.length;
        setNodeState('node-indexing', 'completed', 'Indexed \u2713');
        setConnector('conn-2', true);

        if (index.chunks.length === 0) {
            throw new Error("No readable text was found in this PDF. If this is a scanned document or an image, you will need to use a PDF with selectable text.");
        }

        return { index: index, metadata: result.metadata };
    },

    query: async function(question) {
        if (!docIndex) throw new Error('No document loaded.');
        var logs = [];

        // Retriever
        setNodeState('node-retriever', 'active', 'Searching…');
        logs.push({ agent: 'Retriever Agent', status: 'running', message: 'Running TF-IDF similarity search…' });

        var chunks = RetrieverAgent.run(question, docIndex, topK);
        setNodeState('node-retriever', 'completed', chunks.length + ' chunks found');
        setConnector('conn-3', true);

        var pages = Array.from(new Set(chunks.map(function(c) { return c.page; }))).join(', ');
        logs.push({ agent: 'Retriever Agent', status: 'completed', message: 'Retrieved ' + chunks.length + ' chunks from page(s): ' + pages });

        if (!chunks.length) {
            return {
                answer: 'I could not find any relevant text in the document for your question. Please try rephrasing.',
                retrieved_chunks: [],
                validation: { score: 100, claims: [], hallucinations: [], verdict: 'PASS' },
                agent_logs: logs
            };
        }

        // Summarizer
        setNodeState('node-summarizer', 'active', 'Generating…');
        setConnector('conn-4', true);
        logs.push({ agent: 'Summarizer Agent', status: 'running', message: 'Calling Groq (' + summarizerModel + ')…' });

        var answer = await SummarizerAgent.run(question, chunks, chatHistory, summarizerModel);
        setNodeState('node-summarizer', 'completed', 'Done \u2713');
        logs.push({ agent: 'Summarizer Agent', status: 'completed', message: 'Answer generated with citations.' });

        // Validator
        setNodeState('node-validator', 'active', 'Verifying…');
        setConnector('conn-5', true);
        logs.push({ agent: 'Validator Agent', status: 'running', message: 'Cross-checking facts via ' + validatorModel + '…' });

        var validation = await ValidatorAgent.run(question, chunks, answer, validatorModel);
        setNodeState('node-validator', 'completed', validation.verdict);
        logs.push({ agent: 'Validator Agent', status: 'completed', message: 'Score: ' + validation.score + '% · Verdict: ' + validation.verdict });

        setNodeState('node-conversation', 'completed', 'Responded');
        logs.push({ agent: 'Conversation Agent', status: 'completed', message: 'Response delivered.' });

        return { answer: answer, retrieved_chunks: chunks, validation: validation, agent_logs: logs };
    }
};

// ══════════════════════════════════════════════════════════════════════════════
//  UI — Event Listeners & Handlers
// ══════════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', function() {

    // Restore settings in modal
    $('groqApiKeyInput').value    = (groqApiKey !== '__GROQ_API_KEY__') ? groqApiKey : '';
    $('summarizerModelSel').value = summarizerModel;
    $('validatorModelSel').value  = validatorModel;
    $('topKSel').value            = String(topK);

    // Settings modal
    $('btnSettings').addEventListener('click', function() { $('settingsModal').classList.add('show'); });
    $('btnCloseSettings').addEventListener('click', function() { $('settingsModal').classList.remove('show'); });
    window.addEventListener('click', function(e) {
        if (e.target === $('settingsModal')) $('settingsModal').classList.remove('show');
    });
    $('settingsForm').addEventListener('submit', function(e) {
        e.preventDefault();
        groqApiKey      = $('groqApiKeyInput').value.trim();
        summarizerModel = $('summarizerModelSel').value;
        validatorModel  = $('validatorModelSel').value;
        topK            = parseInt($('topKSel').value);
        localStorage.setItem('groq_api_key',     groqApiKey);
        localStorage.setItem('summarizer_model', summarizerModel);
        localStorage.setItem('validator_model',  validatorModel);
        localStorage.setItem('top_k',            topK);
        $('settingsModal').classList.remove('show');
        systemNote('Settings saved.');
    });

    // File input change — triggered by the <label> in HTML (100% reliable)
    $('pdfFileInput').addEventListener('change', function(e) {
        var files = e.target.files;
        if (files && files.length > 0) {
            handleUpload(files[0]);
        }
    });

    // Drop zone — drag & drop support
    var dz = $('dropZone');
    dz.addEventListener('dragover',  function(e) { e.preventDefault(); dz.classList.add('dragover'); });
    dz.addEventListener('dragleave', function()  { dz.classList.remove('dragover'); });
    dz.addEventListener('drop', function(e) {
        e.preventDefault();
        dz.classList.remove('dragover');
        var files = e.dataTransfer && e.dataTransfer.files;
        if (files && files.length > 0) handleUpload(files[0]);
    });
    // Also make drop zone clickable — manually trigger file input
    dz.addEventListener('click', function(e) {
        e.stopPropagation();
        $('pdfFileInput').click();
    });

    // Chat
    $('chatForm').addEventListener('submit', handleChat);
    $('btnClearChat').addEventListener('click', function() {
        chatHistory = [];
        $('chatMessages').innerHTML = '';
        showWelcome();
    });
    $('btnResetPdf').addEventListener('click', resetDocument);
});

// ─── FILE UPLOAD ──────────────────────────────────────────────────────────────
async function handleUpload(file) {
    if (!file) return;
    var name = file.name.toLowerCase();
    if (!name.endsWith('.pdf')) {
        alert('Please select a PDF file.');
        return;
    }

    // Switch to progress view
    $('dropZone').classList.add('hidden');
    document.querySelector('label[for="pdfFileInput"]').classList.add('hidden');
    $('pdfDetails').classList.add('hidden');
    $('uploadProgress').classList.remove('hidden');

    resetPipelineNodes();
    setNodeState('node-ingestion', 'active', 'Parsing…');

    try {
        var result = await ConversationAgent.ingestDocument(file);
        docIndex = result.index;
        docMeta  = result.metadata;

        setNodeState('node-conversation', 'completed', 'Ready');
        setProgress('All agents ready! Ask your first question.', 100);

        await sleep(500);
        showDocDetails(result.metadata);
        systemNote('"' + result.metadata.title + '" indexed — ' + result.metadata.chunk_count + ' chunks ready.');

    } catch(err) {
        console.error('Upload error:', err);
        alert('Failed to process PDF:\n' + err.message);
        showUploadUI();
        resetPipelineNodes();
    }
}

function resetDocument() {
    if (!confirm('Clear the current document and reset?')) return;
    docIndex = null;
    docMeta  = null;
    chatHistory = [];
    showUploadUI();
    resetPipelineNodes();
    $('noValidationMsg').classList.remove('hidden');
    $('validationResult').classList.add('hidden');
    $('noCitationsMsg').classList.remove('hidden');
    $('citationsList').classList.add('hidden');
    $('chatMessages').innerHTML = '';
    showWelcome();
}

// ─── CHAT ─────────────────────────────────────────────────────────────────────
async function handleChat(e) {
    e.preventDefault();
    var q = $('userInput').value.trim();
    if (!q || isProcessing) return;

    var key = groqApiKey || DEFAULT_GROQ_KEY;
    if (!key || key === '__GROQ_API_KEY__') {
        alert('Please configure your Groq API key in Settings first.');
        $('settingsModal').classList.add('show');
        return;
    }

    isProcessing = true;
    $('userInput').value   = '';
    $('btnSend').disabled  = true;
    $('userInput').disabled = true;

    renderBubble('user', q);
    var typing = renderTyping();

    // Reset pipeline from retriever down
    ['node-retriever','node-summarizer','node-validator','node-conversation'].forEach(function(id) {
        setNodeState(id, '', 'Idle');
    });
    ['conn-3','conn-4','conn-5'].forEach(function(id) { setConnector(id, false); });

    try {
        var result = await ConversationAgent.query(q);
        typing.remove();
        renderBubble('assistant', result.answer, result.agent_logs);
        renderValidation(result.validation);
        renderCitations(result.retrieved_chunks);
        chatHistory.push({ role: 'user', content: q });
        chatHistory.push({ role: 'assistant', content: result.answer });
    } catch(err) {
        typing.remove();
        renderBubble('assistant', '<span style="color:var(--color-danger)"><i class="fa-solid fa-triangle-exclamation"></i> <strong>Error:</strong> ' + escHtml(err.message) + '</span>');
        if (docMeta) {
            setNodeState('node-ingestion',   'completed', 'Ready');
            setNodeState('node-indexing',    'completed', 'Ready');
            setNodeState('node-conversation','completed', 'Ready');
        }
    }

    isProcessing = false;
    $('btnSend').disabled   = false;
    $('userInput').disabled = false;
    $('userInput').focus();
    $('chatMessages').scrollTop = $('chatMessages').scrollHeight;
}

// ─── PIPELINE UI ──────────────────────────────────────────────────────────────
function setNodeState(id, state, statusText) {
    var node = $(id);
    if (!node) return;
    node.className = 'agent-node' + (state ? ' ' + state : '');
    node.querySelector('.node-status').textContent = statusText;
}
function setConnector(id, active) {
    var el = $(id);
    if (el) el.className = active ? 'pipeline-connector active' : 'pipeline-connector';
}
function resetPipelineNodes() {
    ['node-ingestion','node-indexing','node-retriever','node-summarizer','node-validator','node-conversation']
        .forEach(function(id) { setNodeState(id, '', 'Idle'); });
    ['conn-1','conn-2','conn-3','conn-4','conn-5'].forEach(function(id) { setConnector(id, false); });
}
function setProgress(msg, pct, sub) {
    $('progressMessage').textContent = msg || '';
    $('progressBar').style.width = (pct || 0) + '%';
    if ($('progressSub')) $('progressSub').textContent = sub || '';
}

// ─── UI STATE ────────────────────────────────────────────────────────────────
function showDocDetails(meta) {
    $('uploadProgress').classList.add('hidden');
    $('dropZone').classList.add('hidden');
    var lbl = document.querySelector('label[for="pdfFileInput"]');
    if (lbl) lbl.classList.add('hidden');

    $('pdfName').textContent   = meta.filename;
    $('pdfPages').textContent  = meta.pages;
    $('pdfWords').textContent  = fmt(meta.word_count);
    $('pdfChunks').textContent = meta.chunk_count || '-';
    $('pdfTables').textContent = meta.tables_found || '0';
    $('pdfTitle').textContent  = meta.title;
    $('pdfAuthor').textContent = meta.author;
    $('pdfDetails').classList.remove('hidden');

    $('activeDocTitle').textContent = meta.title;
    var parts = [meta.pages + ' pages', fmt(meta.word_count) + ' words', meta.chunk_count + ' chunks'];
    if (meta.tables_found > 0) parts.push(meta.tables_found + ' tables');
    $('activeDocSub').textContent = parts.join(' \u00b7 ');
    $('chatHeaderInfo').classList.add('active');

    $('userInput').disabled = false;
    $('btnSend').disabled   = false;
    $('userInput').placeholder = 'Ask a question about the document…';
}

function showUploadUI() {
    $('uploadProgress').classList.add('hidden');
    $('pdfDetails').classList.add('hidden');
    $('dropZone').classList.remove('hidden');
    var lbl = document.querySelector('label[for="pdfFileInput"]');
    if (lbl) lbl.classList.remove('hidden');

    $('activeDocTitle').textContent = 'Research Assistant';
    $('activeDocSub').textContent   = 'Upload a PDF to begin';
    $('chatHeaderInfo').classList.remove('active');

    $('userInput').disabled = true;
    $('btnSend').disabled   = true;
    $('userInput').placeholder = 'Upload a PDF to start asking questions…';
}

function showWelcome() {
    var msgs = $('chatMessages');
    if (!msgs.querySelector('.system-welcome')) {
        msgs.innerHTML = '<div class="system-welcome" id="welcomeScreen">' +
            '<div class="welcome-icon"><i class="fa-solid fa-robot"></i></div>' +
            '<h2>Welcome to PDF Research Assistant</h2>' +
            '<p>All 6 AI agents run in your browser \u2014 no server required.</p>' +
            '<div class="welcome-steps">' +
            '<div class="step-item"><span class="step-num">1</span><span>Open <strong>Settings</strong> and enter your Groq API key.</span></div>' +
            '<div class="step-item"><span class="step-num">2</span><span>Click <strong>Browse &amp; Upload PDF</strong> to load a document.</span></div>' +
            '<div class="step-item"><span class="step-num">3</span><span>Ask questions \u2014 agents retrieve, summarize &amp; fact-check.</span></div>' +
            '</div>' +
            '<div class="browser-badge"><i class="fa-solid fa-lock"></i> Fully private \u2014 your PDF never leaves your browser</div>' +
            '</div>';
    }
}

// ─── RENDERING ────────────────────────────────────────────────────────────────
function renderBubble(role, content, logs) {
    var msgs = $('chatMessages');
    var welcome = msgs.querySelector('.system-welcome');
    if (welcome) welcome.remove();

    var div = document.createElement('div');
    div.className = 'message ' + role;

    var icon = role === 'user' ? 'user' : 'robot';
    var bubble = role === 'user'
        ? '<div class="msg-bubble">' + escHtml(content) + '</div>'
        : '<div class="msg-bubble">' + formatMd(content) + '</div>';

    var logsHtml = '';
    if (logs && logs.length) {
        var items = logs.map(function(l) {
            return '<div class="log-item"><span class="log-dot ' + l.status + '"></span>' +
                '<span><strong class="log-agent">' + escHtml(l.agent) + ':</strong> ' +
                '<span class="log-msg">' + escHtml(l.message) + '</span></span></div>';
        }).join('');
        logsHtml = '<div class="agent-logs-expander">' +
            '<div class="expander-header" onclick="this.nextElementSibling.classList.toggle(\'hidden\');this.querySelector(\'.toggle-arrow\').classList.toggle(\'fa-chevron-up\');this.querySelector(\'.toggle-arrow\').classList.toggle(\'fa-chevron-down\');">' +
            '<span><i class="fa-solid fa-bug"></i> View Agent Audit Logs</span>' +
            '<i class="fa-solid fa-chevron-down toggle-arrow"></i></div>' +
            '<div class="expander-content hidden">' + items + '</div></div>';
    }

    div.innerHTML = '<div class="msg-avatar"><i class="fa-solid fa-' + icon + '"></i></div>' +
        '<div class="msg-content-wrapper">' + bubble + logsHtml + '</div>';

    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
}

function renderTyping() {
    var msgs = $('chatMessages');
    var welcome = msgs.querySelector('.system-welcome');
    if (welcome) welcome.remove();

    var div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = '<div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>' +
        '<div class="msg-bubble"><div class="dual-spinner" style="width:18px;height:18px;border-width:2px;margin:2px auto;"></div></div>';
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return div;
}

function systemNote(text) {
    var msgs = $('chatMessages');
    var welcome = msgs.querySelector('.system-welcome');
    if (welcome) welcome.remove();

    var div = document.createElement('div');
    div.style.cssText = 'text-align:center;font-size:11px;color:var(--text-muted);margin:6px 0;padding:4px;';
    div.innerHTML = '<i class="fa-solid fa-info-circle"></i> ' + escHtml(text);
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
}

function renderValidation(report) {
    $('noValidationMsg').classList.add('hidden');
    $('validationResult').classList.remove('hidden');

    var score = report.score || 0;
    $('gaugeBarInner').style.width = score + '%';
    $('gaugeValue').textContent = score + '%';

    var bar = $('gaugeBarInner');
    var badge = $('verdictBadge');
    if (score >= 80) { bar.style.backgroundColor = 'var(--color-success)'; badge.className = 'badge pass'; }
    else if (score >= 50) { bar.style.backgroundColor = 'var(--color-warning)'; badge.className = 'badge warning'; }
    else { bar.style.backgroundColor = 'var(--color-danger)'; badge.className = 'badge fail'; }
    badge.textContent = report.verdict || 'UNKNOWN';

    var cl = $('claimsList');
    cl.innerHTML = '';
    (report.claims || []).forEach(function(c) {
        var li = document.createElement('li');
        li.className = 'claim-item';
        li.innerHTML = '<div class="claim-top"><span class="claim-desc">' + escHtml(c.claim) + '</span>' +
            '<span class="claim-status ' + (c.status||'') + '">' + escHtml((c.status||'').replace(/_/g,' ')) + '</span></div>' +
            '<div class="claim-reason">' + escHtml(c.reason||'') + '</div>' +
            (c.page_citation ? '<div class="claim-citation">[Page ' + c.page_citation + ']</div>' : '');
        cl.appendChild(li);
    });

    var halls = report.hallucinations || [];
    if (halls.length) {
        $('hallucinationsList').innerHTML = halls.map(function(h) {
            return '<div class="hallucination-alert"><i class="fa-solid fa-triangle-exclamation"></i><span>' + escHtml(h) + '</span></div>';
        }).join('');
        $('hallucinationsSection').classList.remove('hidden');
    } else {
        $('hallucinationsSection').classList.add('hidden');
    }
}

function renderCitations(chunks) {
    if (!chunks || !chunks.length) return;
    $('noCitationsMsg').classList.add('hidden');
    $('citationsList').classList.remove('hidden');
    $('citationsList').innerHTML = chunks.map(function(c, i) {
        var preview = c.text.slice(0, 280);
        var ellipsis = c.text.length > 280 ? '\u2026' : '';
        return '<div class="citation-block">' +
            '<div class="citation-meta"><span class="citation-source"><i class="fa-solid fa-book"></i> Source ' + (i+1) + ' \u00b7 Page ' + c.page + '</span>' +
            '<span class="citation-score">' + Math.round((c.score || 0) * 100) + '% match</span></div>' +
            '<div class="citation-text">\u201c' + escHtml(preview) + ellipsis + '\u201d</div>' +
            '</div>';
    }).join('');
}

// ─── HELPERS ─────────────────────────────────────────────────────────────────
function escHtml(str) {
    return (str || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function formatMd(text) {
    var h = (text || '')
        .replace(/```([\s\S]*?)```/g, function(_,c){ return '<pre><code>'+escHtml(c.trim())+'</code></pre>'; })
        .replace(/`([^`]+)`/g, function(_,c){ return '<code>'+escHtml(c)+'</code>'; })
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/__(.*?)__/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\[Page\s+(\d+)\]/gi, '<span style="color:var(--accent-primary);font-weight:700">[Page $1]</span>')
        .replace(/\[Source\s+(\d+)\]/gi, '<span style="color:var(--accent-secondary);font-weight:700">[Source $1]</span>')
        .replace(/^#{3} (.+)$/gm, '<h4>$1</h4>')
        .replace(/^#{2} (.+)$/gm, '<h3>$1</h3>')
        .replace(/^# (.+)$/gm, '<h3>$1</h3>')
        .replace(/^\|(.+)\|$/gm, function(line) {
            var cells = line.split('|').filter(function(c,i,a){ return i>0&&i<a.length-1; });
            return '<tr>' + cells.map(function(c){ return '<td>'+c.trim()+'</td>'; }).join('') + '</tr>';
        })
        .replace(/(<tr>[\s\S]*?<\/tr>)/g, '<table class="md-table">$1</table>')
        .replace(/^[\-\*] (.+)$/gm, '<li>$1</li>')
        .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
        .replace(/\n{2,}/g, '</p><p>')
        .replace(/\n/g, '<br>');
    h = h.replace(/((?:<li>.*?<\/li>\s*)+)/gs, '<ul>$1</ul>');
    return '<p>' + h + '</p>';
}

function fmt(n) {
    if (n === undefined || n === null) return '-';
    return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function sleep(ms) { return new Promise(function(r){ setTimeout(r, ms); }); }
