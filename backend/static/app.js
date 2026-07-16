// Global state variables
let sessionId = "default";
let chatHistory = [];
let documentLoaded = false;

// Configs (saved in localStorage)
let groqApiKey = localStorage.getItem("groq_api_key") || "";
let summarizerModel = localStorage.getItem("summarizer_model") || "llama-3.1-8b-instant";
let validatorModel = localStorage.getItem("validator_model") || "llama-3.3-70b-versatile";

// DOM Elements
const doc = {
    // Header
    connectionStatus: document.getElementById("connectionStatus"),
    btnSettings: document.getElementById("btnSettings"),
    
    // Ingestion sidebar
    dropZone: document.getElementById("dropZone"),
    pdfFileInput: document.getElementById("pdfFileInput"),
    uploadProgress: document.getElementById("uploadProgress"),
    progressBar: document.getElementById("progressBar"),
    progressMessage: document.getElementById("progressMessage"),
    pdfDetails: document.getElementById("pdfDetails"),
    pdfName: document.getElementById("pdfName"),
    pdfPages: document.getElementById("pdfPages"),
    pdfWords: document.getElementById("pdfWords"),
    pdfTitle: document.getElementById("pdfTitle"),
    pdfAuthor: document.getElementById("pdfAuthor"),
    btnResetPdf: document.getElementById("btnResetPdf"),
    btnBrowse: document.getElementById("btnBrowse"),
    
    // Pipeline Nodes
    nodeIngestion: document.getElementById("node-ingestion"),
    nodeIndexing: document.getElementById("node-indexing"),
    nodeRetriever: document.getElementById("node-retriever"),
    nodeSummarizer: document.getElementById("node-summarizer"),
    nodeValidator: document.getElementById("node-validator"),
    nodeConversation: document.getElementById("node-conversation"),
    
    // Pipeline Connectors
    conn1: document.getElementById("conn-1"),
    conn2: document.getElementById("conn-2"),
    conn3: document.getElementById("conn-3"),
    conn4: document.getElementById("conn-4"),
    conn5: document.getElementById("conn-5"),
    
    // Chat workspace
    activeDocTitle: document.getElementById("activeDocTitle"),
    activeDocSub: document.querySelector(".active-doc-sub"),
    chatMessages: document.getElementById("chatMessages"),
    chatForm: document.getElementById("chatForm"),
    userInput: document.getElementById("userInput"),
    btnSend: document.getElementById("btnSend"),
    btnClearChat: document.getElementById("btnClearChat"),
    
    // Right panel: Validation
    noValidationMsg: document.getElementById("noValidationMsg"),
    validationResult: document.getElementById("validationResult"),
    gaugeBarInner: document.getElementById("gaugeBarInner"),
    gaugeValue: document.getElementById("gaugeValue"),
    verdictBadge: document.getElementById("verdictBadge"),
    claimsList: document.getElementById("claimsList"),
    hallucinationsSection: document.getElementById("hallucinationsSection"),
    hallucinationsList: document.getElementById("hallucinationsList"),
    
    // Right panel: Citations
    noCitationsMsg: document.getElementById("noCitationsMsg"),
    citationsList: document.getElementById("citationsList"),
    
    // Modal
    settingsModal: document.getElementById("settingsModal"),
    btnCloseSettings: document.getElementById("btnCloseSettings"),
    settingsForm: document.getElementById("settingsForm"),
    groqApiKeyInput: document.getElementById("groqApiKey"),
    summarizerModelSelect: document.getElementById("summarizerModel"),
    validatorModelSelect: document.getElementById("validatorModel")
};

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
    init();
});

function init() {
    // 1. Load settings from storage
    if (groqApiKey) {
        doc.groqApiKeyInput.value = groqApiKey;
    }
    doc.summarizerModelSelect.value = summarizerModel;
    doc.validatorModelSelect.value = validatorModel;
    
    // Show settings modal on startup if API key is missing
    if (!groqApiKey) {
        openSettingsModal();
    }
    
    // 2. Fetch server status (checks if PDF was already loaded previously)
    checkServerStatus();
    
    // 3. Attach Event Listeners
    setupEventListeners();
}

function checkServerStatus() {
    fetch(`/api/status?session_id=${sessionId}`)
        .then(res => res.json())
        .then(data => {
            setConnectedState(true);
            if (data.status === "indexed" && data.metadata) {
                showDocumentDetails(data.metadata);
            } else {
                showUploadState();
            }
        })
        .catch(err => {
            console.error("Failed to connect to server:", err);
            setConnectedState(false);
        });
}

function setConnectedState(isConnected) {
    if (isConnected) {
        doc.connectionStatus.className = "connection-status connected";
        doc.connectionStatus.innerHTML = '<span class="status-dot"></span> Server Online';
    } else {
        doc.connectionStatus.className = "connection-status disconnected";
        doc.connectionStatus.innerHTML = '<span class="status-dot"></span> Server Offline';
    }
}

function setupEventListeners() {
    // Settings modal triggers
    doc.btnSettings.addEventListener("click", openSettingsModal);
    doc.btnCloseSettings.addEventListener("click", closeSettingsModal);
    window.addEventListener("click", (e) => {
        if (e.target === doc.settingsModal) {
            closeSettingsModal();
        }
    });
    
    // Save settings form
    doc.settingsForm.addEventListener("submit", (e) => {
        e.preventDefault();
        groqApiKey = doc.groqApiKeyInput.value.trim();
        summarizerModel = doc.summarizerModelSelect.value;
        validatorModel = doc.validatorModelSelect.value;
        
        localStorage.setItem("groq_api_key", groqApiKey);
        localStorage.setItem("summarizer_model", summarizerModel);
        localStorage.setItem("validator_model", validatorModel);
        
        closeSettingsModal();
        appendSystemLog("System", "Configuration saved successfully.");
    });
    
    // Upload: browse button triggers file input
    doc.btnBrowse.addEventListener("click", (e) => {
        e.stopPropagation();
        // Reset value so same file can be re-selected
        doc.pdfFileInput.value = "";
        doc.pdfFileInput.click();
    });
    
    doc.pdfFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
    
    // Drag & Drop on the drop zone (keep visual drop working)
    doc.dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.stopPropagation();
        doc.dropZone.classList.add("dragover");
    });
    doc.dropZone.addEventListener("dragleave", (e) => {
        e.stopPropagation();
        doc.dropZone.classList.remove("dragover");
    });
    doc.dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        doc.dropZone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
    
    // Clear Document trigger
    doc.btnResetPdf.addEventListener("click", handleResetSession);
    
    // Chat Submissions
    doc.chatForm.addEventListener("submit", handleChatMessage);
    
    // Clear Chat logs
    doc.btnClearChat.addEventListener("click", () => {
        chatHistory = [];
        doc.chatMessages.innerHTML = "";
        showWelcomeScreen();
    });
}

function openSettingsModal() {
    doc.settingsModal.classList.add("show");
}

function closeSettingsModal() {
    doc.settingsModal.classList.remove("show");
}

/* --- PIPELINE VISUAL ANIMATIONS --- */
function resetPipelineNodes() {
    const nodes = [doc.nodeIngestion, doc.nodeIndexing, doc.nodeRetriever, doc.nodeSummarizer, doc.nodeValidator, doc.nodeConversation];
    const conns = [doc.conn1, doc.conn2, doc.conn3, doc.conn4, doc.conn5];
    
    nodes.forEach(node => {
        node.className = "agent-node";
        node.querySelector(".node-status").textContent = "Idle";
    });
    
    conns.forEach(conn => {
        conn.className = "pipeline-connector";
    });
}

function setPipelineNodeState(node, state, statusText) {
    if (!node) return;
    
    node.className = `agent-node ${state}`; // state: 'active', 'completed', or '' (idle)
    node.querySelector(".node-status").textContent = statusText;
}

function setConnectorState(conn, isActive) {
    if (!conn) return;
    conn.className = isActive ? "pipeline-connector active" : "pipeline-connector";
}

/* --- FILE UPLOAD LOGIC --- */
function handleFileUpload(file) {
    if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
        alert("Please upload a valid PDF document.");
        return;
    }
    
    // Start visual ingestion progress
    doc.dropZone.classList.add("hidden");
    doc.pdfDetails.classList.add("hidden");
    doc.uploadProgress.classList.remove("hidden");
    
    resetPipelineNodes();
    
    // Step 1: Ingestion Agent starts
    setPipelineNodeState(doc.nodeIngestion, "active", "Processing text...");
    doc.progressBar.style.width = "25%";
    doc.progressMessage.textContent = "Ingestion Agent: Extracting PDF pages & metadata...";
    
    const formData = new FormData();
    formData.append("file", file);
    
    fetch(`/api/upload?session_id=${sessionId}`, {
        method: "POST",
        body: formData
    })
    .then(async response => {
        // Step 2: Indexing Agent starts (mock transition for server-side processing duration)
        setPipelineNodeState(doc.nodeIngestion, "completed", "Parsed");
        setConnectorState(doc.conn1, true);
        setPipelineNodeState(doc.nodeIndexing, "active", "Encoding chunks...");
        doc.progressBar.style.width = "65%";
        doc.progressMessage.textContent = "Indexing Agent: Building semantic vector map...";
        
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Failed to process PDF file.");
        }
        return data;
    })
    .then(data => {
        // Step 3: Pipeline completed
        setPipelineNodeState(doc.nodeIndexing, "completed", "Indexed");
        setConnectorState(doc.conn2, true);
        setPipelineNodeState(doc.nodeConversation, "completed", "Active");
        doc.progressBar.style.width = "100%";
        doc.progressMessage.textContent = "Retrieval pipeline active!";
        
        setTimeout(() => {
            showDocumentDetails(data.metadata);
            appendSystemLog("System", `Document "${data.metadata.title}" has been successfully parsed and cached.`);
            
            // Clear message inputs
            doc.userInput.disabled = false;
            doc.btnSend.disabled = false;
            doc.userInput.placeholder = "Ask a question about the uploaded document...";
            doc.userInput.focus();
        }, 800);
    })
    .catch(err => {
        console.error("Upload error:", err);
        alert(`Ingestion failed: ${err.message}`);
        showUploadState();
        resetPipelineNodes();
    });
}

function showDocumentDetails(metadata) {
    doc.uploadProgress.classList.add("hidden");
    doc.dropZone.classList.add("hidden");
    doc.btnBrowse.classList.add("hidden"); // hide browse button while doc is loaded
    
    doc.pdfName.textContent = metadata.filename;
    doc.pdfPages.textContent = metadata.pages;
    doc.pdfWords.textContent = formatNumber(metadata.word_count);
    doc.pdfTitle.textContent = metadata.title;
    doc.pdfAuthor.textContent = metadata.author;
    
    // Show extra content info if available
    let subtitleParts = [`${metadata.pages} pages`, `${formatNumber(metadata.word_count)} words`];
    if (metadata.tables_found > 0) subtitleParts.push(`${metadata.tables_found} tables`);
    if (metadata.images_found > 0) subtitleParts.push(`${metadata.images_found} images`);
    
    doc.pdfDetails.classList.remove("hidden");
    
    // Update Chatroom Header info
    doc.activeDocTitle.textContent = metadata.title;
    doc.activeDocSub.textContent = subtitleParts.join(" · ");
    document.querySelector(".active-chat-info").classList.add("active");
    
    // Enable Chat inputs
    doc.userInput.disabled = false;
    doc.btnSend.disabled = false;
    doc.userInput.placeholder = "Ask a question about the uploaded document...";
    
    documentLoaded = true;
    
    // Set static completed nodes for Sidebar Pipeline
    resetPipelineNodes();
    setPipelineNodeState(doc.nodeIngestion, "completed", "Ready");
    setPipelineNodeState(doc.nodeIndexing, "completed", "Ready");
    setPipelineNodeState(doc.nodeConversation, "completed", "Ready");
}

function showUploadState() {
    doc.uploadProgress.classList.add("hidden");
    doc.pdfDetails.classList.add("hidden");
    doc.dropZone.classList.remove("hidden");
    doc.btnBrowse.classList.remove("hidden"); // show browse button again
    
    // Update Chatroom Header info
    doc.activeDocTitle.textContent = "Research Chatroom";
    doc.activeDocSub.textContent = "No document loaded";
    document.querySelector(".active-chat-info").classList.remove("active");
    
    // Disable Chat inputs
    doc.userInput.value = "";
    doc.userInput.disabled = true;
    doc.btnSend.disabled = true;
    doc.userInput.placeholder = "Upload a PDF document to begin...";
    
    documentLoaded = false;
}

function handleResetSession() {
    if (!confirm("Are you sure you want to unload the current document and reset the server cache?")) {
        return;
    }
    
    fetch(`/api/reset?session_id=${sessionId}`, { method: "POST" })
        .then(res => res.json())
        .then(() => {
            showUploadState();
            resetPipelineNodes();
            resetValidationPanel();
            resetCitationsPanel();
            doc.chatMessages.innerHTML = "";
            showWelcomeScreen();
            appendSystemLog("System", "Cached document index has been cleared.");
        })
        .catch(err => {
            console.error("Reset error:", err);
            alert("Failed to reset session server-side.");
        });
}

/* --- CHAT FLOW & API INTEGRATION --- */
function handleChatMessage(e) {
    e.preventDefault();
    
    const message = doc.userInput.value.trim();
    if (!message) return;
    
    if (!groqApiKey) {
        alert("Please set your Groq API Key in the settings modal first!");
        openSettingsModal();
        return;
    }
    
    // 1. Add User bubble to chat list
    renderMessageBubble("user", message);
    doc.userInput.value = "";
    
    // Create temporary typing bubble
    const typingBubble = renderTypingBubble();
    
    // 2. Animate Agent Pipeline - Retriever starts
    resetPipelineNodes();
    setPipelineNodeState(doc.nodeIngestion, "completed", "Ready");
    setPipelineNodeState(doc.nodeIndexing, "completed", "Ready");
    
    setPipelineNodeState(doc.nodeRetriever, "active", "Searching database...");
    setConnectorState(doc.conn2, true);
    
    // 3. Make Server API query
    const requestData = {
        message: message,
        session_id: sessionId,
        api_key: groqApiKey,
        history: chatHistory,
        summarizer_model: summarizerModel,
        validator_model: validatorModel
    };
    
    fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData)
    })
    .then(async res => {
        const responseText = await res.text();
        let data;
        try {
            data = JSON.parse(responseText);
        } catch (jsonErr) {
            throw new Error(`Server returned non-JSON payload: ${responseText.substring(0, 100)}`);
        }
        
        if (!res.ok) {
            throw new Error(data.detail || "Error querying document.");
        }
        return data;
    })
    .then(data => {
        // Stepwise updates for final logs
        // Retriever finishes -> Summarizer starts
        setPipelineNodeState(doc.nodeRetriever, "completed", "Success");
        setConnectorState(doc.conn3, true);
        
        setPipelineNodeState(doc.nodeSummarizer, "active", "Summarizing...");
        
        // Mock brief visual transition
        setTimeout(() => {
            setPipelineNodeState(doc.nodeSummarizer, "completed", "Done");
            setConnectorState(doc.conn4, true);
            setPipelineNodeState(doc.nodeValidator, "active", "Verifying claims...");
            
            setTimeout(() => {
                setPipelineNodeState(doc.nodeValidator, "completed", data.validation.verdict || "PASS");
                setConnectorState(doc.conn5, true);
                setPipelineNodeState(doc.nodeConversation, "completed", "Responded");
                
                // Clear typing indicator and render final answer
                typingBubble.remove();
                
                // Render Assistant bubble
                renderMessageBubble("assistant", data.answer, data.agent_logs);
                
                // Update panels
                renderValidationReport(data.validation);
                renderCitations(data.retrieved_chunks);
                
                // Save to history
                chatHistory.push({ role: "user", content: message });
                chatHistory.push({ role: "assistant", content: data.answer });
                
                // Scroll to bottom
                doc.chatMessages.scrollTop = doc.chatMessages.scrollHeight;
            }, 600);
        }, 600);
    })
    .catch(err => {
        console.error("Chat error:", err);
        typingBubble.remove();
        resetPipelineNodes();
        
        setPipelineNodeState(doc.nodeConversation, "active", "Error");
        renderMessageBubble("assistant", `<span style="color:var(--color-danger);"><i class="fa-solid fa-triangle-exclamation"></i> <strong>Orchestration Error:</strong> ${err.message}</span>`);
    });
}

/* --- RENDER HELPER FUNCTIONS --- */
function renderMessageBubble(role, content, agentLogs = null) {
    // Clear welcome message if present
    const welcome = doc.chatMessages.querySelector(".system-welcome");
    if (welcome) welcome.remove();
    
    const msgElement = document.createElement("div");
    msgElement.className = `message ${role}`;
    
    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.innerHTML = role === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
    
    const wrapper = document.createElement("div");
    wrapper.className = "msg-content-wrapper";
    
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    
    // Parse formatting (light markdown parsing: code blocks, lists, citations, headers)
    bubble.innerHTML = formatMarkdown(content);
    
    wrapper.appendChild(bubble);
    
    // If agent execution logs exist, append the collapsible audit drawer
    if (agentLogs && agentLogs.length > 0) {
        const expander = document.createElement("div");
        expander.className = "agent-logs-expander";
        
        const header = document.createElement("div");
        header.className = "expander-header";
        header.innerHTML = '<span><i class="fa-solid fa-bug"></i> View Agent Audit Logs</span> <i class="fa-solid fa-chevron-down toggle-arrow"></i>';
        
        const contentDiv = document.createElement("div");
        contentDiv.className = "expander-content hidden";
        
        agentLogs.forEach(log => {
            const logItem = document.createElement("div");
            logItem.className = "log-item";
            
            const dot = document.createElement("span");
            dot.className = `log-dot ${log.status}`;
            
            const text = document.createElement("span");
            text.innerHTML = `<strong class="log-agent">${log.agent}:</strong> <span class="log-msg">${log.message}</span>`;
            
            logItem.appendChild(dot);
            logItem.appendChild(text);
            contentDiv.appendChild(logItem);
        });
        
        // Collapsible event
        header.addEventListener("click", () => {
            contentDiv.classList.toggle("hidden");
            const icon = header.querySelector(".toggle-arrow");
            if (contentDiv.classList.contains("hidden")) {
                icon.className = "fa-solid fa-chevron-down toggle-arrow";
            } else {
                icon.className = "fa-solid fa-chevron-up toggle-arrow";
            }
        });
        
        expander.appendChild(header);
        expander.appendChild(contentDiv);
        wrapper.appendChild(expander);
    }
    
    msgElement.appendChild(avatar);
    msgElement.appendChild(wrapper);
    
    doc.chatMessages.appendChild(msgElement);
    doc.chatMessages.scrollTop = doc.chatMessages.scrollHeight;
}

function renderTypingBubble() {
    // Clear welcome message
    const welcome = doc.chatMessages.querySelector(".system-welcome");
    if (welcome) welcome.remove();
    
    const msgElement = document.createElement("div");
    msgElement.className = "message assistant";
    
    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.innerHTML = '<i class="fa-solid fa-robot"></i>';
    
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.innerHTML = '<div class="dual-spinner" style="width:20px; height:20px; border-width:2px;"></div>';
    
    msgElement.appendChild(avatar);
    msgElement.appendChild(bubble);
    doc.chatMessages.appendChild(msgElement);
    doc.chatMessages.scrollTop = doc.chatMessages.scrollHeight;
    
    return msgElement;
}

function showWelcomeScreen() {
    doc.chatMessages.innerHTML = `
        <div class="system-welcome">
            <div class="welcome-icon"><i class="fa-solid fa-robot"></i></div>
            <h2>Welcome to PDF Research Assistant</h2>
            <p>To begin, please follow these steps:</p>
            <div class="welcome-steps">
                <div class="step-item">
                    <span class="step-num">1</span>
                    <span>Open the <strong>Settings</strong> button in the top right to configure your <strong>Groq API Key</strong>.</span>
                </div>
                <div class="step-item">
                    <span class="step-num">2</span>
                    <span>Upload your PDF research paper or technical document in the left sidebar.</span>
                </div>
                <div class="step-item">
                    <span class="step-num">3</span>
                    <span>Ask questions, extract summaries, and verify the model's factual assertions in real-time.</span>
                </div>
            </div>
        </div>
    `;
}

function appendSystemLog(type, text) {
    const wel = doc.chatMessages.querySelector(".system-welcome");
    if (wel) wel.remove();
    
    const div = document.createElement("div");
    div.style.textAlign = "center";
    div.style.fontSize = "11px";
    div.style.color = "var(--text-muted)";
    div.style.margin = "8px 0";
    div.innerHTML = `<i class="fa-solid fa-info-circle"></i> [${type}] ${text}`;
    doc.chatMessages.appendChild(div);
    doc.chatMessages.scrollTop = doc.chatMessages.scrollHeight;
}

/* --- SIDEBAR REPORTS RENDERING --- */
function renderValidationReport(report) {
    doc.noValidationMsg.classList.add("hidden");
    doc.validationResult.classList.remove("hidden");
    
    const score = report.score || 0;
    doc.gaugeValue.textContent = `${score}%`;
    doc.gaugeBarInner.style.width = `${score}%`;
    
    // Adjust colors based on score
    if (score >= 80) {
        doc.gaugeBarInner.style.backgroundColor = "var(--color-success)";
        doc.verdictBadge.className = "badge pass";
    } else if (score >= 50) {
        doc.gaugeBarInner.style.backgroundColor = "var(--color-warning)";
        doc.verdictBadge.className = "badge warning";
    } else {
        doc.gaugeBarInner.style.backgroundColor = "var(--color-danger)";
        doc.verdictBadge.className = "badge fail";
    }
    doc.verdictBadge.textContent = report.verdict || "FAIL";
    
    // Render Claims
    doc.claimsList.innerHTML = "";
    if (report.claims && report.claims.length > 0) {
        report.claims.forEach(c => {
            const item = document.createElement("li");
            item.className = "claim-item";
            
            const citeText = c.page_citation ? `[Page ${c.page_citation}]` : "";
            const statusLabel = c.status.replace("_", " ");
            
            item.innerHTML = `
                <div class="claim-top">
                    <span class="claim-desc">${c.claim}</span>
                    <span class="claim-status ${c.status}">${statusLabel}</span>
                </div>
                <div class="claim-reason">${c.reason}</div>
                ${citeText ? `<div class="claim-citation">${citeText}</div>` : ""}
            `;
            doc.claimsList.appendChild(item);
        });
    } else {
        doc.claimsList.innerHTML = '<li class="panel-placeholder" style="padding:10px;"><p>No granular claims evaluated.</p></li>';
    }
    
    // Render Hallucinations
    if (report.hallucinations && report.hallucinations.length > 0) {
        doc.hallucinationsList.innerHTML = "";
        report.hallucinations.forEach(h => {
            const alertDiv = document.createElement("div");
            alertDiv.className = "hallucination-alert";
            alertDiv.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <span>${h}</span>`;
            doc.hallucinationsList.appendChild(alertDiv);
        });
        doc.hallucinationsSection.classList.remove("hidden");
    } else {
        doc.hallucinationsSection.classList.add("hidden");
    }
}

function resetValidationPanel() {
    doc.noValidationMsg.classList.remove("hidden");
    doc.validationResult.classList.add("hidden");
    doc.claimsList.innerHTML = "";
    doc.hallucinationsList.innerHTML = "";
    doc.gaugeBarInner.style.width = "0%";
    doc.gaugeValue.textContent = "0%";
}

function renderCitations(chunks) {
    doc.noCitationsMsg.classList.add("hidden");
    doc.citationsList.classList.remove("hidden");
    
    doc.citationsList.innerHTML = "";
    
    if (chunks && chunks.length > 0) {
        chunks.forEach((chunk, index) => {
            const div = document.createElement("div");
            div.className = "citation-block";
            
            const confidence = Math.round(chunk.score * 100);
            
            div.innerHTML = `
                <div class="citation-meta">
                    <span class="citation-source"><i class="fa-solid fa-book"></i> Source ${index + 1} (Page ${chunk.page})</span>
                    <span class="citation-score">Relevance: ${confidence}%</span>
                </div>
                <div class="citation-text">"${chunk.text}"</div>
            `;
            doc.citationsList.appendChild(div);
        });
    } else {
        doc.citationsList.innerHTML = '<div class="panel-placeholder"><p>No references found.</p></div>';
    }
}

function resetCitationsPanel() {
    doc.noCitationsMsg.classList.remove("hidden");
    doc.citationsList.classList.add("hidden");
    doc.citationsList.innerHTML = "";
}

/* --- TEXT FORMATTING UTIL (MARKDOWN) --- */
function formatMarkdown(text) {
    if (!text) return "";
    
    let html = text;
    
    // Escape HTML symbols first to protect content, except formatting constructs
    html = html.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    
    // Code blocks
    html = html.replace(/```([\s\S]*?)```/g, (match, p1) => {
        return `<pre><code>${p1.trim()}</code></pre>`;
    });
    
    // Inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    
    // Bold tags
    html = html.replace(/\*\*([\s\S]*?)\*\*/g, "<strong>$1</strong>");
    
    // Citations styling like [Page X] or [Source Y]
    html = html.replace(/\[(Page\s+\d+)\]/gi, '<span style="color:var(--accent-primary); font-weight:700;">[$1]</span>');
    html = html.replace(/\[(Source\s+\d+)\]/gi, '<span style="color:var(--accent-secondary); font-weight:700;">[$1]</span>');
    
    // Line breaks
    html = html.split("\n").map(line => {
        // Bullet list items
        if (line.trim().startsWith("- ") || line.trim().startsWith("* ")) {
            return `<ul><li>${line.trim().substring(2)}</li></ul>`;
        }
        // Ordered list items
        const numMatch = line.trim().match(/^(\d+)\.\s+(.*)/);
        if (numMatch) {
            return `<ol start="${numMatch[1]}"><li>${numMatch[2]}</li></ol>`;
        }
        // Headers
        if (line.trim().startsWith("### ")) {
            return `<h4>${line.trim().substring(4)}</h4>`;
        }
        if (line.trim().startsWith("## ")) {
            return `<h3>${line.trim().substring(3)}</h3>`;
        }
        if (line.trim().startsWith("# ")) {
            return `<h2>${line.trim().substring(2)}</h2>`;
        }
        
        return line ? `<p>${line}</p>` : "";
    }).join("");
    
    // Merge consecutive list blocks
    html = html.replace(/<\/ul><ul>/g, "").replace(/<\/ol><ol[^>]*>/g, "");
    
    return html;
}

function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
