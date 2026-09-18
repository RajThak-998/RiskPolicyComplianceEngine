const state = { sessionId: null, selectedFile: null, poller: null, busy: false };
const $ = (id) => document.getElementById(id);

function showToast(message) {
  const toast = $('toast');
  toast.textContent = message;
  toast.classList.add('show');
  window.setTimeout(() => toast.classList.remove('show'), 3200);
}

function setService(online) {
  $('serviceStatus').textContent = online ? 'Engine online' : 'Engine unavailable';
  document.querySelector('.status-dot').classList.toggle('online', online);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

async function startSession() {
  try {
    const session = await api('/v1/sessions', { method: 'POST' });
    state.sessionId = session.session_id;
    setService(true);
  } catch (error) {
    setService(false);
    showToast(error.message);
  }
}

function chooseFile() { $('fileInput').click(); }
function selectFile(file) {
  if (!file) return;
  if (file.type !== 'application/pdf') { showToast('Please choose a PDF file.'); return; }
  state.selectedFile = file;
  $('titleInput').value = file.name.replace(/\.pdf$/i, '').replace(/[_-]+/g, ' ');
  $('uploadForm').hidden = false;
  $('dropzone').querySelector('strong').textContent = file.name;
  $('dropzone').querySelector('span').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB selected`;
}

function renderDocuments(documents = []) {
  $('documents').innerHTML = documents.map((doc) => {
    const detail = doc.status === 'ready' ? `${doc.pages} pages / ${doc.chunks} chunks` : (doc.error || doc.status);
    return `<div class="document"><div class="document-title">${escapeHtml(doc.title)}</div><div class="document-meta"><span>${escapeHtml(detail)}</span><span class="${doc.status}">${escapeHtml(doc.status)}</span></div></div>`;
  }).join('');
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (char) => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', "'":'&#39;', '"':'&quot;' }[char]));
}

async function refreshStatus() {
  if (!state.sessionId) return;
  try {
    const session = await api(`/v1/sessions/${state.sessionId}/status`);
    renderDocuments(session.documents);
    if (session.status === 'ready' && session.documents.length) {
      $('emptyState').hidden = true;
      $('questionForm').hidden = false;
      $('cacheBadge').textContent = 'Ready to audit';
      $('cacheBadge').classList.add('hit');
      stopPolling();
    } else if (session.status === 'failed') {
      $('cacheBadge').textContent = 'Processing failed';
      stopPolling();
      showToast('The policy could not be indexed. Check the document and try again.');
    }
  } catch (error) { showToast(error.message); stopPolling(); }
}

function startPolling() {
  stopPolling();
  refreshStatus();
  state.poller = window.setInterval(refreshStatus, 1200);
}
function stopPolling() { if (state.poller) window.clearInterval(state.poller); state.poller = null; }

async function upload(event) {
  event.preventDefault();
  if (!state.selectedFile || !state.sessionId) return;
  const form = new FormData();
  form.append('file', state.selectedFile);
  form.append('title', $('titleInput').value.trim());
  form.append('policy_type', $('typeInput').value.trim() || 'Unknown');
  const button = $('uploadForm').querySelector('button');
  button.disabled = true;
  button.textContent = 'Indexing...';
  try {
    await api(`/v1/sessions/${state.sessionId}/documents`, { method: 'POST', body: form });
    $('cacheBadge').textContent = 'Indexing evidence';
    $('questionForm').hidden = true;
    $('emptyState').hidden = false;
    $('emptyState').querySelector('h2').textContent = 'Indexing your evidence';
    $('emptyState').querySelector('p').textContent = 'Retrieval and citation checks will unlock when the document is ready.';
    $('uploadForm').hidden = true;
    startPolling();
  } catch (error) {
    showToast(error.message);
    button.disabled = false;
    button.innerHTML = 'Index policy <span>-></span>';
  }
}

function addMessage(type, html) {
  const messages = $('messages');
  messages.hidden = false;
  const item = document.createElement('div');
  item.className = `message ${type}`;
  item.innerHTML = `<div class="message-label">${type === 'user' ? 'You' : 'Policy desk'}</div><div class="message-bubble">${html}</div>`;
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
  return item.querySelector('.message-bubble');
}

function renderVerdict(payload, target) {
  const verdict = payload.verdict || {};
  const statusClass = verdict.status === 'OUT_OF_SCOPE' ? 'out' : 'covered';
  const citations = (verdict.citations || []).map((citation) => `<div class="citation"><b>${escapeHtml(citation.document_id)} / p.${escapeHtml(citation.page_number)} / ${escapeHtml(citation.clause_id)}</b><br>"${escapeHtml(citation.exact_quote)}"</div>`).join('');
  const failures = (payload.grounding_failures || []).map(escapeHtml).join('<br>');
  target.innerHTML = `<div class="verdict"><div class="verdict-head"><span class="verdict-status ${statusClass}">${escapeHtml(verdict.status || 'NO VERDICT')}</span><span class="message-label">${payload.cached ? 'Redis cache hit' : 'Fresh audit'}</span></div><div class="verdict-facts"><div class="fact"><label>Financial limit</label><strong>${escapeHtml(verdict.financial_limit || 'Not stated')}</strong></div><div class="fact"><label>Deductible</label><strong>${escapeHtml(verdict.applicable_deductible || 'Not stated')}</strong></div></div><div class="verdict-reasoning">${escapeHtml(verdict.reasoning || 'No reasoning returned.')}</div>${citations ? `<div class="citations"><div class="message-label">Verified sources</div>${citations}</div>` : ''}${failures ? `<div class="grounding">Grounding warning: ${failures}</div>` : ''}</div>`;
}

async function consumeAudit(query, target) {
  const response = await fetch('/v1/audit/stream', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ query, session_id:state.sessionId, jurisdiction:'Unknown', effective_year:2024, policy_type:'Unknown', stream:true }) });
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `Audit failed (${response.status})`); }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalPayload = null;
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const events = buffer.split('\n\n');
    buffer = events.pop() || '';
    for (const raw of events) {
      const dataLine = raw.split('\n').find((line) => line.startsWith('data: '));
      if (!dataLine) continue;
      const data = JSON.parse(dataLine.slice(6));
      if (raw.startsWith('event: verdict')) { finalPayload = data; renderVerdict(data, target); $('cacheBadge').textContent = data.cached ? 'Redis cache hit' : 'Fresh audit'; $('cacheBadge').classList.toggle('hit', data.cached); }
      if (raw.startsWith('event: error')) throw new Error(data.message || 'The audit stream failed.');
    }
    if (done) break;
  }
  if (!finalPayload) throw new Error('The audit ended without a verdict.');
}

async function ask(event) {
  event.preventDefault();
  if (state.busy) return;
  const input = $('questionInput');
  const query = input.value.trim();
  if (!query) return;
  state.busy = true;
  input.value = '';
  addMessage('user', escapeHtml(query));
  const target = addMessage('assistant', '<span class="message-label">Retrieving evidence and auditing...</span>');
  try { await consumeAudit(query, target); }
  catch (error) { target.innerHTML = `<span style="color:#bd4e3e">${escapeHtml(error.message)}</span>`; showToast(error.message); }
  finally { state.busy = false; }
}

$('chooseFile').addEventListener('click', chooseFile);
$('fileInput').addEventListener('change', (event) => selectFile(event.target.files[0]));
$('uploadForm').addEventListener('submit', upload);
$('questionForm').addEventListener('submit', ask);
$('dropzone').addEventListener('dragover', (event) => { event.preventDefault(); $('dropzone').classList.add('dragging'); });
$('dropzone').addEventListener('dragleave', () => $('dropzone').classList.remove('dragging'));
$('dropzone').addEventListener('drop', (event) => { event.preventDefault(); $('dropzone').classList.remove('dragging'); selectFile(event.dataTransfer.files[0]); });
startSession();
