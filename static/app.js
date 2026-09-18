/**
 * Policy Desk — App Logic
 *
 * Flow:
 *  1. Page load → POST /v1/sessions → get sessionId
 *  2. User selects PDF, fills title/type, hits "Index policy"
 *  3. POST /v1/sessions/:id/documents (multipart) → 202 accepted
 *  4. Poll /v1/sessions/:id/status every 1.2 s → animate pipeline steps
 *  5. When status === "ready" → show ready-overlay, enable query form
 *  6. User submits query → SSE stream from /v1/audit/stream
 *     - cache HIT  → single "verdict" event → render instantly + show REDIS HIT badge
 *     - cache MISS → series of "token" events (stream text) + final "verdict" event
 *     - hallucination detected → show warning banner in message
 */

/* ─── Helpers ─────────────────────────────────────────────────── */
const $ = (id) => document.getElementById(id);

const state = {
  sessionId: null,
  selectedFile: null,
  poller: null,
  busy: false,
  ingestionStart: null,
};

function escapeHtml(v) {
  return String(v ?? '').replace(/[&<>'"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c])
  );
}

function showToast(msg, type = 'error') {
  const t = $('toast');
  t.textContent = msg;
  t.className = `toast show ${type}`;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove('show'), 4000);
}

function setService(online) {
  $('serviceStatus').textContent = online ? 'Engine online' : 'Engine unavailable';
  $('statusDot').classList.toggle('online', online);
}

/* ─── API helper ─────────────────────────────────────────────── */
async function api(path, options = {}) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

/* ─── Step bar ───────────────────────────────────────────────── */
function setStep(n) {
  [1, 2, 3].forEach((i) => {
    const el = $(`step${i}`);
    el.classList.remove('active', 'done');
    if (i < n) el.classList.add('done');
    if (i === n) el.classList.add('active');
  });
}

/* ─── Pipeline steps ─────────────────────────────────────────── */
const PIPELINE_STEPS = ['ps-upload', 'ps-chunk', 'ps-embed', 'ps-index', 'ps-ready'];
let pipelineTimer = null;

function setPipelineStep(stepId, status) {
  const el = $(stepId);
  if (!el) return;
  const icon = el.querySelector('.ps-icon');
  // Always clear all state classes first so they never pile up
  el.classList.remove('done', 'active', 'error');
  if (status === 'done') {
    el.classList.add('done');
    icon.textContent = '✓';
  } else if (status === 'active') {
    el.classList.add('active');
    icon.textContent = '⟳';
  } else if (status === 'error') {
    el.classList.add('error');
    icon.textContent = '✕';
  }
}

function startPipelineAnimation() {
  // Step 1 (upload) is instantly done since we got 202
  setPipelineStep('ps-upload', 'done');
  setPipelineStep('ps-chunk', 'active');

  let phase = 1; // 0=upload done, 1=chunk, 2=embed, 3=index, 4=ready
  pipelineTimer = setInterval(() => {
    if (phase === 1) {
      setPipelineStep('ps-chunk', 'done');
      setPipelineStep('ps-embed', 'active');
      phase++;
    } else if (phase === 2) {
      setPipelineStep('ps-embed', 'done');
      setPipelineStep('ps-index', 'active');
      phase++;
    } else if (phase === 3) {
      setPipelineStep('ps-index', 'done');
      setPipelineStep('ps-ready', 'active');
      phase++;
    }
    // phase 4 = wait for real poll to fire
  }, 3500); // cosmetic pacing; real signal comes from poll
}

async function finishPipeline(success) {
  clearInterval(pipelineTimer);
  pipelineTimer = null;

  // Sweep through any steps not yet marked done, one at a time (300 ms apart)
  // so the user can see them tick off sequentially before "complete" appears.
  const pending = PIPELINE_STEPS.filter((id) => {
    const el = $(id);
    return el && !el.classList.contains('done');
  });

  for (const id of pending) {
    await new Promise((r) => setTimeout(r, 300));
    setPipelineStep(id, success ? 'done' : 'error');
  }

  $('pipelineStatus').querySelector('.pipeline-title').textContent =
    success ? '✓ Pipeline complete' : '✕ Pipeline failed';
  $('pipelineStatus').classList.toggle('pipeline-failed', !success);
}

/* ─── Session ─────────────────────────────────────────────────── */
async function startSession() {
  try {
    const session = await api('/v1/sessions', { method: 'POST' });
    state.sessionId = session.session_id;
    setService(true);
    setStep(1);
  } catch (err) {
    setService(false);
    showToast(err.message);
  }
}

/* ─── File selection ─────────────────────────────────────────── */
function selectFile(file) {
  if (!file) return;
  if (file.type !== 'application/pdf') {
    showToast('Please choose a PDF file.');
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    showToast('File exceeds 25 MB limit.');
    return;
  }
  state.selectedFile = file;
  $('dropzoneTitle').textContent = file.name;
  $('dropzoneSub').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB selected`;
  $('titleInput').value = file.name.replace(/\.pdf$/i, '').replace(/[_-]+/g, ' ');
  $('uploadForm').hidden = false;
  $('dropzone').classList.add('file-selected');
}

/* ─── Document list ──────────────────────────────────────────── */
function renderDocuments(documents = []) {
  $('documents').innerHTML = documents
    .map((doc) => {
      const detail =
        doc.status === 'ready'
          ? `${doc.pages ?? '?'} pages · ${doc.chunks ?? '?'} chunks`
          : doc.error || doc.status;
      return `
        <div class="document">
          <div class="document-title">${escapeHtml(doc.title)}</div>
          <div class="document-meta">
            <span>${escapeHtml(detail)}</span>
            <span class="doc-status ${doc.status}">${escapeHtml(doc.status)}</span>
          </div>
        </div>`;
    })
    .join('');
}

/* ─── Status polling ─────────────────────────────────────────── */
async function refreshStatus() {
  if (!state.sessionId) return;
  try {
    const session = await api(`/v1/sessions/${state.sessionId}/status`);
    renderDocuments(session.documents);

    if (session.status === 'ready' && session.documents.length) {
      stopPolling();                 // stop poll first so it won't re-enter
      await finishPipeline(true);   // sweep all steps to ✓, then resolve
      onIngestionComplete();        // THEN show overlay + unlock chat
    } else if (session.status === 'failed') {
      finishPipeline(false);
      stopPolling();
      showToast('Ingestion failed. Check the document and try again.');
    }
  } catch (err) {
    showToast(err.message);
    stopPolling();
  }
}

function startPolling() {
  stopPolling();
  state.poller = setInterval(refreshStatus, 1200);
}

function stopPolling() {
  if (state.poller) clearInterval(state.poller);
  state.poller = null;
}

/* ─── Ingestion complete → show overlay ─────────────────────── */
function onIngestionComplete() {
  setStep(3);
  $('readyOverlay').hidden = false;
  // update empty-state text (it may still be visible behind overlay)
  $('emptyState').hidden = true;
  $('messages').hidden = false;
  $('questionForm').hidden = false;
  setCacheBadge('ready');
}

/* ─── Upload ──────────────────────────────────────────────────── */
async function upload(event) {
  event.preventDefault();
  if (!state.selectedFile || !state.sessionId) return;

  const btn = $('uploadBtn');
  btn.disabled = true;
  btn.innerHTML = 'Indexing… <span>⟳</span>';

  const form = new FormData();
  form.append('file', state.selectedFile);
  form.append('title', $('titleInput').value.trim());
  form.append('policy_type', $('typeInput').value.trim() || 'Unknown');

  try {
    await api(`/v1/sessions/${state.sessionId}/documents`, { method: 'POST', body: form });

    // Reveal pipeline UI
    $('uploadForm').hidden = true;
    $('pipelineStatus').hidden = false;
    $('emptyState').querySelector('h2').textContent = 'Indexing your evidence…';
    $('emptyState').querySelector('p').textContent =
      'Chunking, embedding, and indexing. The chat unlocks when everything is ready.';
    setStep(2);
    setCacheBadge('indexing');
    startPipelineAnimation();
    startPolling();

    state.ingestionStart = Date.now();
    showToast('PDF accepted — indexing in background.', 'info');
  } catch (err) {
    showToast(err.message);
    btn.disabled = false;
    btn.innerHTML = 'Index policy <span>→</span>';
  }
}

/* ─── Cache badge ─────────────────────────────────────────────── */
function setCacheBadge(mode, distanceMs) {
  const badge = $('cacheBadge');
  badge.className = 'cache-badge';
  switch (mode) {
    case 'ready':
      badge.textContent = 'Ready to audit';
      badge.classList.add('badge-ready');
      break;
    case 'indexing':
      badge.textContent = 'Indexing evidence…';
      badge.classList.add('badge-indexing');
      break;
    case 'hit':
      badge.innerHTML = '⚡ Redis cache hit';
      badge.classList.add('badge-hit');
      if (distanceMs !== undefined)
        badge.title = `Cosine distance: ${distanceMs.toFixed(4)}`;
      break;
    case 'miss':
      badge.innerHTML = '⟳ Cache miss — live RAG';
      badge.classList.add('badge-miss');
      break;
    default:
      badge.textContent = 'Waiting for policy';
  }
}

/* ─── Messages ────────────────────────────────────────────────── */
function addMessage(type, html) {
  const wrap = $('messages');
  wrap.hidden = false;
  const div = document.createElement('div');
  div.className = `message ${type}`;
  div.innerHTML = `
    <div class="message-label">${type === 'user' ? 'You' : 'Policy Desk'}</div>
    <div class="message-bubble">${html}</div>`;
  wrap.appendChild(div);
  wrap.scrollTop = wrap.scrollHeight;
  return div.querySelector('.message-bubble');
}

/* ─── Verdict rendering ───────────────────────────────────────── */
function renderVerdict(payload, target) {
  const v = payload.verdict || {};
  const isOos = v.status === 'OUT_OF_SCOPE';
  const statusClass = isOos ? 'out' : 'covered';

  // Citations
  const citationsHtml = (v.citations || [])
    .map(
      (c) => `
      <div class="citation">
        <b>${escapeHtml(c.document_id)} / p.${escapeHtml(c.page_number)} / ${escapeHtml(c.clause_id)}</b><br>
        <q>${escapeHtml(c.exact_quote)}</q>
      </div>`
    )
    .join('');

  // Hallucination / grounding failures
  const failures = payload.grounding_failures || [];
  const hallucinationHtml =
    failures.length > 0
      ? `<div class="hallucination-warning">
           <span class="hall-icon">⚠</span>
           <div>
             <strong>Hallucination detected</strong>
             <ul>${failures.map((f) => `<li>${escapeHtml(f)}</li>`).join('')}</ul>
           </div>
         </div>`
      : '';

  // Cache / source label
  const sourceLabel = payload.cached
    ? `<span class="source-tag tag-cached">⚡ Redis cache hit · dist ${(payload.cache_distance || 0).toFixed(3)}</span>`
    : `<span class="source-tag tag-rag">⟳ Live RAG · fresh generation</span>`;

  target.innerHTML = `
    <div class="verdict">
      <div class="verdict-head">
        <span class="verdict-status ${statusClass}">${escapeHtml(v.status || 'NO VERDICT')}</span>
        ${sourceLabel}
      </div>
      <div class="verdict-facts">
        <div class="fact">
          <label>Financial limit</label>
          <strong>${escapeHtml(v.financial_limit || 'Not stated')}</strong>
        </div>
        <div class="fact">
          <label>Deductible</label>
          <strong>${escapeHtml(v.applicable_deductible || 'Not stated')}</strong>
        </div>
      </div>
      <div class="verdict-reasoning">${escapeHtml(v.reasoning || 'No reasoning returned.')}</div>
      ${citationsHtml ? `<div class="citations"><div class="citations-label">Verified sources</div>${citationsHtml}</div>` : ''}
      ${hallucinationHtml}
    </div>`;
}

/* ─── Streaming token rendering ───────────────────────────────── */
function createStreamingBubble() {
  const wrap = $('messages');
  wrap.hidden = false;
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.innerHTML = `
    <div class="message-label">Policy Desk</div>
    <div class="message-bubble">
      <div class="stream-header">
        <span class="stream-badge">⟳ Generating…</span>
      </div>
      <div class="stream-raw" id="streamRaw"></div>
      <span class="stream-cursor">▋</span>
    </div>`;
  wrap.appendChild(div);
  wrap.scrollTop = wrap.scrollHeight;
  return div.querySelector('.message-bubble');
}

/* ─── Consume SSE audit stream ───────────────────────────────── */
async function consumeAudit(query, target) {
  const body = JSON.stringify({
    query,
    session_id: state.sessionId,
    jurisdiction: 'Unknown',
    effective_year: 2024,
    policy_type: 'Unknown',
    stream: true,
  });

  const res = await fetch('/v1/audit/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Audit failed (${res.status})`);
  }

  // Detect if cache hit from response headers early
  const cacheHeader = res.headers.get('X-Cache'); // 'HIT' or 'MISS'
  const isCacheHit = cacheHeader === 'HIT';

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let receivedVerdict = false;
  let streamRawEl = null; // the raw token accumulator element
  let cursorEl = null;

  // If it's a cache miss, show a streaming bubble right away
  if (!isCacheHit) {
    target.innerHTML = `
      <div class="stream-header">
        <span class="stream-badge">⟳ Fetching from knowledge base…</span>
      </div>
      <div class="stream-raw" id="streamRaw"></div>
      <span class="stream-cursor">▋</span>`;
    streamRawEl = document.getElementById('streamRaw');
    cursorEl = target.querySelector('.stream-cursor');
    setCacheBadge('miss');
  } else {
    target.innerHTML = `<span class="message-label">Retrieving from Redis…</span>`;
    setCacheBadge('hit');
  }

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    const events = buffer.split('\n\n');
    buffer = events.pop() || '';

    for (const raw of events) {
      const eventLine = raw.split('\n').find((l) => l.startsWith('event: '));
      const dataLine = raw.split('\n').find((l) => l.startsWith('data: '));
      if (!dataLine) continue;

      const eventType = eventLine ? eventLine.slice(7).trim() : 'message';
      let data;
      try {
        data = JSON.parse(dataLine.slice(6));
      } catch {
        continue;
      }

      if (eventType === 'meta') {
        // Candidates count
        const count = (data.candidates || []).length;
        if (streamRawEl) {
          const header = target.querySelector('.stream-badge');
          if (header)
            header.textContent = `⟳ Generating — ${count} candidate chunk${count !== 1 ? 's' : ''} retrieved…`;
        }
        setInputHint(`${count} chunks retrieved · reranked to top ${Math.min(count, 5)}`);
      }

      if (eventType === 'token') {
        // Append token text to the raw stream area
        if (streamRawEl) {
          streamRawEl.textContent += data.token || '';
          $('messages').scrollTop = $('messages').scrollHeight;
        }
      }

      if (eventType === 'verdict') {
        receivedVerdict = true;
        // Remove cursor + streaming header, render structured verdict
        if (cursorEl) cursorEl.remove();
        renderVerdict(data, target);
        setCacheBadge(data.cached ? 'hit' : 'miss', data.cache_distance);

        // Show hallucination toast if there were failures
        if ((data.grounding_failures || []).length > 0) {
          showToast(
            `⚠ ${data.grounding_failures.length} grounding failure(s) — hallucination risk flagged.`,
            'warn'
          );
        }
      }

      if (eventType === 'error') {
        throw new Error(data.message || 'Audit stream error.');
      }
    }

    if (done) break;
  }

  if (!receivedVerdict) {
    throw new Error('Audit ended without a verdict.');
  }

  setInputHint('');
}

/* ─── Input hint ─────────────────────────────────────────────── */
function setInputHint(msg) {
  const el = $('inputHint');
  if (!el) return;
  el.textContent = msg;
  el.style.display = msg ? 'block' : 'none';
}

/* ─── Ask question ───────────────────────────────────────────── */
async function ask(event) {
  event.preventDefault();
  if (state.busy) return;

  const input = $('questionInput');
  const query = input.value.trim();
  if (!query) return;

  state.busy = true;
  $('sendBtn').disabled = true;
  $('sendBtn').innerHTML = '… <span>⟳</span>';
  input.value = '';
  setInputHint('');

  addMessage('user', escapeHtml(query));
  const target = addMessage('assistant', '<span class="message-label">Checking cache and indexing…</span>');

  try {
    await consumeAudit(query, target);
  } catch (err) {
    target.innerHTML = `<span class="error-text">✕ ${escapeHtml(err.message)}</span>`;
    showToast(err.message);
  } finally {
    state.busy = false;
    $('sendBtn').disabled = false;
    $('sendBtn').innerHTML = 'Send <span>→</span>';
  }
}

/* ─── Ready overlay dismiss ──────────────────────────────────── */
$('readyDismiss').addEventListener('click', () => {
  $('readyOverlay').hidden = true;
  $('questionInput').focus();
  showToast('Policy is ready. Ask your first question below!', 'info');
});

/* ─── Drag & drop ────────────────────────────────────────────── */
const dz = $('dropzone');
dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('dragging'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragging'));
dz.addEventListener('drop', (e) => {
  e.preventDefault();
  dz.classList.remove('dragging');
  selectFile(e.dataTransfer.files[0]);
});

/* ─── Wire up events ─────────────────────────────────────────── */
$('chooseFile').addEventListener('click', () => $('fileInput').click());
$('fileInput').addEventListener('change', (e) => selectFile(e.target.files[0]));
$('uploadForm').addEventListener('submit', upload);
$('questionForm').addEventListener('submit', ask);

// Ctrl+Enter to submit
$('questionInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    $('questionForm').requestSubmit();
  }
});

/* ─── Boot ───────────────────────────────────────────────────── */
startSession();
