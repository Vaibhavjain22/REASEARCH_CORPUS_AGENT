/* ════════════════════════════════════════════════════════════
   Research Corpus Agent — Frontend Application Logic
   ════════════════════════════════════════════════════════════ */

// ── DOM References ───────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const navBtns        = $$('.nav-btn');
const tabPanels      = $$('.tab-panel');
const searchInput    = $('#search-input');
const searchBtn      = $('#search-btn');
const chips          = $$('.chip');
const loadingState   = $('#loading-state');
const resultContainer = $('#result-container');
const resultContent  = $('#result-content');
const resultQuery    = $('#result-query-label');
const resultTime     = $('#result-time');
const welcomeState   = $('#welcome-state');
const historyList    = $('#history-list');
const ingestForm     = $('#ingest-form');
const ingestStatus   = $('#ingest-status');
const ingestBtn      = $('#ingest-submit-btn');
const mobileMenuBtn  = $('#mobile-menu-btn');
const sidebar        = $('#sidebar');
const sidebarOverlay = $('#sidebar-overlay');

// ── Agent step cycling ───────────────────────────────────────
const agentSteps     = ['step-planner', 'step-retriever', 'step-analyst', 'step-critic'];
let agentInterval    = null;


// ══════════════════════════════════════════════════════════════
//  TAB NAVIGATION
// ══════════════════════════════════════════════════════════════

navBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        navBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        tabPanels.forEach((p) => p.classList.remove('active'));
        $(`#tab-${tab}`).classList.add('active');

        // Load evaluation data on first visit
        if (tab === 'evaluation' && !window._evalLoaded) {
            loadEvaluationData();
            window._evalLoaded = true;
        }

        // Close mobile sidebar
        closeMobileSidebar();
    });
});


// ══════════════════════════════════════════════════════════════
//  MOBILE SIDEBAR
// ══════════════════════════════════════════════════════════════

mobileMenuBtn.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    sidebarOverlay.classList.toggle('open');
});

sidebarOverlay.addEventListener('click', closeMobileSidebar);

function closeMobileSidebar() {
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('open');
}


// ══════════════════════════════════════════════════════════════
//  SEARCH
// ══════════════════════════════════════════════════════════════

searchBtn.addEventListener('click', executeSearch);
searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') executeSearch();
});

chips.forEach((chip) => {
    chip.addEventListener('click', () => {
        searchInput.value = chip.dataset.query;
        executeSearch();
    });
});

async function executeSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    // Show loading, hide others
    welcomeState.classList.add('hidden');
    resultContainer.classList.add('hidden');
    loadingState.classList.remove('hidden');
    searchBtn.disabled = true;

    startAgentCycling();

    try {
        const res = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });

        const data = await res.json();

        stopAgentCycling();
        loadingState.classList.add('hidden');
        searchBtn.disabled = false;

        if (res.ok) {
            showResult(data.query, data.answer, data.time_seconds);
            loadHistory();   // refresh sidebar
            searchInput.value = '';
        } else {
            showResult(query, `**Error:** ${data.error || 'Something went wrong.'}`, 0);
        }
    } catch (err) {
        stopAgentCycling();
        loadingState.classList.add('hidden');
        searchBtn.disabled = false;
        showResult(query, `**Error:** Network error — ${err.message}`, 0);
    }
}

function showResult(query, answer, timeSec) {
    resultQuery.textContent = query;
    resultTime.textContent = timeSec > 0 ? `⏱ ${timeSec}s` : '';
    resultContent.innerHTML = renderMarkdown(answer);
    resultContainer.classList.remove('hidden');
}


// ── Agent pipeline animation ─────────────────────────────────

function startAgentCycling() {
    let idx = 0;
    setActiveAgent(0);

    agentInterval = setInterval(() => {
        idx++;
        if (idx < agentSteps.length) {
            setActiveAgent(idx);
        } else {
            // loop back to keep it alive
            idx = 0;
            agentSteps.forEach((id) => {
                const el = document.getElementById(id);
                el.classList.remove('active', 'done');
            });
            setActiveAgent(0);
        }
    }, 8000);  // advance every 8s to roughly match typical agent timing

    // Also cycle loading messages
    const messages = [
        'Planner is decomposing your query…',
        'Retriever is searching the knowledge base…',
        'Analyst is synthesizing insights…',
        'Critic is validating the answer…',
    ];
    let msgIdx = 0;
    $('#loading-message').textContent = messages[0];
    window._msgInterval = setInterval(() => {
        msgIdx = (msgIdx + 1) % messages.length;
        $('#loading-message').textContent = messages[msgIdx];
    }, 8000);
}

function setActiveAgent(idx) {
    agentSteps.forEach((id, i) => {
        const el = document.getElementById(id);
        el.classList.remove('active', 'done');
        if (i < idx) el.classList.add('done');
        if (i === idx) el.classList.add('active');
    });
}

function stopAgentCycling() {
    clearInterval(agentInterval);
    clearInterval(window._msgInterval);
    agentSteps.forEach((id) => {
        const el = document.getElementById(id);
        el.classList.remove('active');
        el.classList.add('done');
    });
}


// ══════════════════════════════════════════════════════════════
//  SIMPLE MARKDOWN RENDERER
// ══════════════════════════════════════════════════════════════

function renderMarkdown(text) {
    if (!text) return '';
    let html = text;

    // Escape HTML (basic)
    html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    // Code blocks
    html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Headers (### before ## before #)
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold & Italic
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Blockquotes
    html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

    // Unordered lists
    html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Ordered lists
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Horizontal rules
    html = html.replace(/^---$/gm, '<hr>');

    // Paragraphs — wrap remaining lines
    html = html.replace(/^(?!<[hupblo]|<\/|<li|<hr)(.*\S.*)$/gm, '<p>$1</p>');

    // Clean double breaks
    html = html.replace(/\n{2,}/g, '\n');

    return html;
}


// ══════════════════════════════════════════════════════════════
//  HISTORY
// ══════════════════════════════════════════════════════════════

async function loadHistory() {
    try {
        const res = await fetch('/api/history');
        const data = await res.json();

        if (!data || data.length === 0) {
            historyList.innerHTML = '<p class="history-empty">No queries yet. Start searching!</p>';
            return;
        }

        historyList.innerHTML = data.map((item) => `
            <div class="history-item" data-answer="${encodeURIComponent(item.answer)}" data-query="${encodeURIComponent(item.query)}" data-time="${item.time_seconds || 0}">
                <div class="history-item-query">${escapeHtml(item.query)}</div>
                <div class="history-item-time">${formatTimestamp(item.timestamp)} · ${item.time_seconds || 0}s</div>
            </div>
        `).join('');

        // Attach click handlers
        historyList.querySelectorAll('.history-item').forEach((el) => {
            el.addEventListener('click', () => {
                const query  = decodeURIComponent(el.dataset.query);
                const answer = decodeURIComponent(el.dataset.answer);
                const time   = parseFloat(el.dataset.time);

                // Switch to search tab
                navBtns.forEach((b) => b.classList.remove('active'));
                $('#nav-search').classList.add('active');
                tabPanels.forEach((p) => p.classList.remove('active'));
                $('#tab-search').classList.add('active');

                welcomeState.classList.add('hidden');
                loadingState.classList.add('hidden');
                showResult(query, answer, time);
                closeMobileSidebar();
            });
        });

    } catch {
        // silently ignore
    }
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function formatTimestamp(ts) {
    if (!ts) return '';
    try {
        const d = new Date(ts);
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    } catch { return ''; }
}


// ══════════════════════════════════════════════════════════════
//  INGESTION HUB
// ══════════════════════════════════════════════════════════════

ingestForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const title    = $('#ingest-title').value.trim();
    const authors  = $('#ingest-authors').value.trim();
    const category = $('#ingest-category').value;
    const date     = $('#ingest-date').value;
    const abstract = $('#ingest-abstract').value.trim();

    if (!title || !abstract) return;

    ingestBtn.disabled = true;
    showIngestStatus('loading', '<i class="fas fa-spinner fa-spin"></i> Ingesting paper…');

    try {
        const res = await fetch('/api/ingest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title,
                authors,
                category,
                published_date: date,
                abstract,
            }),
        });

        const data = await res.json();
        ingestBtn.disabled = false;

        if (res.ok) {
            showIngestStatus('success',
                `<i class="fas fa-check-circle"></i> ${data.message} (${data.chunks_added} chunks added, total: ${data.total_documents})`
            );
            ingestForm.reset();
        } else {
            showIngestStatus('error', `<i class="fas fa-times-circle"></i> ${data.error}`);
        }
    } catch (err) {
        ingestBtn.disabled = false;
        showIngestStatus('error', `<i class="fas fa-times-circle"></i> Network error: ${err.message}`);
    }
});

function showIngestStatus(type, html) {
    ingestStatus.className = `ingest-status ${type}`;
    ingestStatus.innerHTML = html;
    ingestStatus.classList.remove('hidden');
}


// ══════════════════════════════════════════════════════════════
//  EVALUATION DASHBOARD
// ══════════════════════════════════════════════════════════════

async function loadEvaluationData() {
    try {
        const res  = await fetch('/api/evaluation');
        const data = await res.json();

        if (data.error) return;

        // ── Summary cards ────────────────────────────────────
        const re = data.retrieval_evaluation;
        const aq = data.answer_quality;

        $('#eval-precision').textContent = `${(re['mean_precision@5'] * 100).toFixed(1)}%`;
        $('#eval-recall').textContent    = `${(re['mean_recall@5'] * 100).toFixed(1)}%`;
        $('#eval-keyword').textContent   = `${(aq.avg_keyword_coverage * 100).toFixed(1)}%`;
        $('#eval-success').textContent   = aq.success_rate;

        // ── Precision & Recall bar chart ─────────────────────
        const queries   = re.per_query.map((q, i) => `Q${i + 1}`);
        const precision = re.per_query.map((q) => q['precision@5']);
        const recall    = re.per_query.map((q) => q['recall@5']);

        new Chart($('#chart-precision-recall'), {
            type: 'bar',
            data: {
                labels: queries,
                datasets: [
                    {
                        label: 'Precision@5',
                        data: precision,
                        backgroundColor: 'hsla(250, 80%, 62%, 0.7)',
                        borderColor: 'hsl(250, 80%, 62%)',
                        borderWidth: 1,
                        borderRadius: 4,
                    },
                    {
                        label: 'Recall@5',
                        data: recall,
                        backgroundColor: 'hsla(152, 62%, 48%, 0.7)',
                        borderColor: 'hsl(152, 62%, 48%)',
                        borderWidth: 1,
                        borderRadius: 4,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: 'hsl(220, 14%, 62%)', font: { size: 11 } } },
                    tooltip: {
                        callbacks: {
                            afterLabel: (ctx) => {
                                const q = re.per_query[ctx.dataIndex];
                                return `Type: ${q.type}\n${q.query.substring(0, 50)}…`;
                            }
                        }
                    }
                },
                scales: {
                    x: { ticks: { color: 'hsl(220, 10%, 42%)', font: { size: 10 } }, grid: { color: 'hsl(228, 16%, 16%)' } },
                    y: { min: 0, max: 1, ticks: { color: 'hsl(220, 10%, 42%)' }, grid: { color: 'hsl(228, 16%, 16%)' } },
                },
            },
        });

        // ── Query type performance doughnut ──────────────────
        const typeMap = {};
        re.per_query.forEach((q) => {
            if (!typeMap[q.type]) typeMap[q.type] = [];
            typeMap[q.type].push(q['precision@5']);
        });

        const typeLabels = Object.keys(typeMap);
        const typeAvgs   = typeLabels.map((t) => {
            const vals = typeMap[t];
            return vals.reduce((a, b) => a + b, 0) / vals.length;
        });

        const doughnutColors = [
            'hsl(250, 80%, 62%)',
            'hsl(152, 62%, 48%)',
            'hsl(38, 92%, 55%)',
            'hsl(210, 90%, 56%)',
            'hsl(0, 72%, 56%)',
        ];

        new Chart($('#chart-query-types'), {
            type: 'doughnut',
            data: {
                labels: typeLabels.map((t) => t.charAt(0).toUpperCase() + t.slice(1)),
                datasets: [{
                    data: typeAvgs,
                    backgroundColor: doughnutColors.slice(0, typeLabels.length),
                    borderColor: 'hsl(228, 24%, 6%)',
                    borderWidth: 3,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: 'hsl(220, 14%, 62%)', font: { size: 11 }, padding: 16 },
                    },
                },
            },
        });

        // ── Failure case list ────────────────────────────────
        const fc = data.failure_case_analysis;
        if (fc && fc.per_case) {
            const failureList = $('#failure-list');
            failureList.innerHTML = fc.per_case.map((c) => `
                <div class="failure-item">
                    <div class="failure-item-query">
                        "${escapeHtml(c.query)}"
                        <span class="failure-badge ${c.handled_well ? 'handled' : 'not-handled'}">
                            ${c.handled_well ? 'Handled' : 'Not Handled'}
                        </span>
                    </div>
                    <div class="failure-item-type">${escapeHtml(c.failure_type)} — ${escapeHtml(c.error_analysis)}</div>
                </div>
            `).join('');
        }

    } catch (err) {
        console.error('Failed to load evaluation data:', err);
    }
}


// ══════════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
});
