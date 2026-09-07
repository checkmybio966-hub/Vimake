/* AI Watermark Remover - front end.
   Talks to the FastAPI backend through RELATIVE urls so it works behind any
   proxy / preview host without CORS or localhost hard-coding. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const state = { items: [], editing: null };
const API = {
  upload: '/api/upload', detect: '/api/detect', process: '/api/process',
  job: id => `/api/jobs/${id}`, backends: '/api/backends',
};

/* ------------------------------------------------------------------ boot */
fetch(API.backends).then(r => r.json()).then(b => {
  $('#backend').textContent = `backend: ${b.active} (${b.available.filter(x => x.ready !== false).map(x => x.name).join(', ')})`;
}).catch(() => { $('#backend').textContent = 'backend: offline'; });

const dz = $('#dropzone');
dz.addEventListener('click', () => $('#fileInput').click());
$('#fileInput').addEventListener('change', e => addFiles([...e.target.files]));
['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('over'); }));
dz.addEventListener('drop', e => addFiles([...(e.dataTransfer.files || [])]));
window.addEventListener('paste', e => addFiles([...(e.clipboardData?.files || [])]));

$('#processAll').addEventListener('click', () => state.items.filter(i => i.info && !i.job).forEach(runJob));
$$('[data-close]').forEach(b => b.addEventListener('click', () => $('#editor').hidden = true));
$('[data-apply]').addEventListener('click', applyMask);

/* --------------------------------------------------------------- upload */
async function addFiles(files) {
  for (const f of files) {
    const item = { id: Math.random().toString(36).slice(2), file: f, name: f.name, info: null, mask: null, mode: 'auto' };
    state.items.push(item);
    render();
    try {
      const fd = new FormData(); fd.append('file', f);
      const r = await fetch(API.upload, { method: 'POST', body: fd });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      item.info = await r.json();
      item.kind = item.info.kind;
      render();
      autoDetect(item);
    } catch (err) {
      item.error = err.message; render();
    }
  }
}

async function autoDetect(item) {
  item.detecting = true; render();
  try {
    const body = new FormData(); body.append('asset_id', item.info.asset_id);
    const r = await fetch(API.detect, { method: 'POST', body });
    const d = await r.json();
    item.det = d;
    if (d.found) { item.maskUrl = d.mask_url; item.overlayUrl = d.overlay_url; }
  } catch (e) { item.det = { found: false, notes: 'detect failed: ' + e.message }; }
  item.detecting = false; render();
}

/* --------------------------------------------------------------- render */
function render() {
  const host = $('#cards'); host.innerHTML = '';
  for (const it of state.items) host.appendChild(card(it));
  $('#processAll').disabled = !state.items.some(i => i.info && !i.job);
}

function card(it) {
  const el = document.createElement('div'); el.className = 'card';
  const det = it.det || {};
  const badge = (it.info?.kind || (it.file?.type?.startsWith('video') ? 'video' : 'image') || '…').toUpperCase();

  let media = `<div class="media"><span class="badge">${badge}</span>`;
  if (it.stage === 'done' && it.resultUrl) {
    if (it.kind === 'video') {
      media += `<div class="compare" style="position:absolute;inset:0">
          <video src="${it.info.source}" muted loop autoplay playsinline></video>
          <video class="after" src="${it.resultUrl}" muted loop autoplay playsinline></video>
          <input type="range" min="0" max="100" value="50" />
          <span class="tag l">original</span><span class="tag r">clean</span>
        </div>`;
    } else {
      media += `<div class="compare" style="position:absolute;inset:0">
          <img src="${it.info.poster}" />
          <img class="after" src="${it.resultUrl}" />
          <input type="range" min="0" max="100" value="50" />
          <span class="tag l">original</span><span class="tag r">clean</span>
        </div>`;
    }
  } else if (it.overlayUrl) {
    media += `<img class="masklayer" src="${it.overlayUrl}" /><img src="${it.info?.poster || it.info?.source || ''}" />`;
  } else if (it.info?.poster) {
    media += `<img src="${it.info.poster}" />`;
  } else if (it.error) {
    media += `<div class="muted">${it.error}</div>`;
  } else {
    media += `<div class="muted">uploading…</div>`;
  }
  media += `</div>`;
  if (it.mask && it.stage !== 'done') {
    media = media.replace(`class="masklayer"`,
      `class="masklayer" style="mix-blend-mode:screen;opacity:.8"`);
  }

  const detPill = it.detecting ? `<span class="pill">detecting…</span>`
    : det.found ? `<span class="pill ok">found ${Math.round((det.score || 0) * 100)}% · ${det.kind} · ${det.method}</span>`
      : it.det ? `<span class="pill warn">nothing detected — brush it</span>` : '';

  let controls = '';
  if (it.stage === 'done') {
    controls = `<a class="btn primary sm" href="${it.resultUrl}" download="${it.resultName || 'clean'}">Download</a>
                <button class="btn sm" data-act="again">Process again</button>`;
  } else if (it.job) {
    controls = `<div class="bar"><i style="width:${Math.round((it.progress || 0) * 100)}%"></i></div>
                <span class="muted" style="font-size:12px">${it.stageText || 'working…'}</span>`;
  } else if (it.info) {
    controls = `<button class="btn sm" data-act="edit">Edit mask</button>
                <button class="btn primary sm" data-act="run">Remove watermark</button>`;
  }

  const modeRow = it.kind === 'video' ? `
    <label class="muted" style="font-size:12px">mark type
      <select data-mode>
        <option value="auto" ${it.mode === 'auto' ? 'selected' : ''}>auto detect</option>
        <option value="static" ${it.mode === 'static' ? 'selected' : ''}>fixed position</option>
        <option value="moving" ${it.mode === 'moving' ? 'selected' : ''}>moving mark</option>
      </select></label>` : '';

  el.innerHTML = `${media}
    <div class="body">
      <div class="name" title="${it.name}">${it.name}</div>
      <div class="meta">${it.info ? meta(it.info) : ''}</div>
      <div class="row">${detPill}${it.det && !det.found && det.notes ? `<span class="pill">${det.notes}</span>` : ''}</div>
      <div class="row">${modeRow}</div>
      <div class="row">${controls}</div>
    </div>`;

  const sel = el.querySelector('[data-mode]');
  if (sel) sel.addEventListener('change', e => { it.mode = e.target.value; render(); });
  el.querySelector('[data-act="edit"]')?.addEventListener('click', () => openEditor(it));
  el.querySelector('[data-act="run"]')?.addEventListener('click', () => runJob(it));
  el.querySelector('[data-act="again"]')?.addEventListener('click', () => { it.job = null; it.stage = null; render(); });
  const range = el.querySelector('.compare input[type=range]');
  if (range) range.addEventListener('input', e => {
    const cmp = el.querySelector('.compare'); cmp.style.setProperty('--split', e.target.value + '%');
  });
  return el;
}

function meta(i) {
  if (i.kind === 'video') return `${i.width}×${i.height} · ${i.fps} fps · ${i.duration}s${i.has_audio ? ' · audio' : ''}`;
  return `${i.width}×${i.height}`;
}

/* ----------------------------------------------------------------- jobs */
async function runJob(it) {
  if (!it.info || it.job) return;
  const fd = new FormData();
  fd.append('asset_id', it.info.asset_id);
  fd.append('mode', it.mode || 'auto');
  if (it.mask) fd.append('mask', it.mask);
  it.job = 'starting'; it.progress = 0; it.stageText = 'queued'; render();
  try {
    const r = await fetch(API.process, { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    it.job = d.job_id; poll(it);
  } catch (e) { it.error = e.message; it.job = null; render(); }
}

async function poll(it) {
  for (let i = 0; i < 3000; i++) {
    const r = await fetch(API.job(it.job)); const s = await r.json();
    it.progress = s.progress || 0; it.stageText = s.stage || s.status;
    if (s.status === 'done') {
      it.stage = 'done'; it.resultUrl = s.result_url; it.resultName = s.result_name;
      it.det = s.detection || it.det; render(); return;
    }
    if (s.status === 'error') { it.error = s.message; it.job = null; render(); return; }
    if (i % 2 === 0) render();
    await new Promise(r2 => setTimeout(r2, 700));
  }
}

/* ----------------------------------------------------------- mask editor */
const canvas = $('#canvas');
const ctx = canvas.getContext('2d');
let layers = null;   // {base, add, sub, auto}
let tool = 'brush', drawing = false, history = [];

$$('.tool').forEach(b => b.addEventListener('click', () => {
  tool = b.dataset.tool; $$('.tool').forEach(x => x.classList.toggle('active', x === b));
}));
$('#undo').addEventListener('click', undo);
$('#clearUser').addEventListener('click', () => {
  if (!layers) return;
  layers.add.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  layers.sub.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  history = []; draw();
});
$('#showAuto').addEventListener('change', draw);
$('#mode').addEventListener('change', e => { if (state.editing) state.editing.mode = e.target.value; });

async function openEditor(it) {
  state.editing = it;
  $('#editor').hidden = false;
  $('#editorName').textContent = it.name;
  $('#modeRow').hidden = it.kind !== 'video';
  $('#mode').value = it.mode || 'auto';

  const img = new Image(); img.crossOrigin = 'anonymous';
  img.src = it.info.poster || it.info.source;
  await img.decode();
  const maxW = 1000;
  const s = Math.min(1, maxW / img.naturalWidth);
  canvas.width = Math.round(img.naturalWidth * s);
  canvas.height = Math.round(img.naturalHeight * s);

  const mk = () => { const c = document.createElement('canvas'); c.width = canvas.width; c.height = canvas.height; return c; };
  layers = { img, auto: mk(), add: mk(), sub: mk() };
  if (it.maskUrl) {
    const m = new Image(); m.src = it.maskUrl;
    try { await m.decode(); layers.auto.getContext('2d').drawImage(m, 0, 0, canvas.width, canvas.height); } catch (e) {}
  }
  history = []; draw();
}

function draw() {
  if (!layers) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(layers.img, 0, 0, canvas.width, canvas.height);
  const tmp = document.createElement('canvas'); tmp.width = canvas.width; tmp.height = canvas.height;
  const t = tmp.getContext('2d');
  if ($('#showAuto').checked) t.drawImage(layers.auto, 0, 0);
  t.globalCompositeOperation = 'source-over';
  t.drawImage(layers.add, 0, 0);
  t.globalCompositeOperation = 'destination-out';
  t.drawImage(layers.sub, 0, 0);
  const red = document.createElement('canvas'); red.width = canvas.width; red.height = canvas.height;
  const rc = red.getContext('2d');
  rc.drawImage(tmp, 0, 0); rc.globalCompositeOperation = 'source-in';
  rc.fillStyle = 'rgba(255,40,80,.55)'; rc.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(red, 0, 0);
}

function pt(e) {
  const r = canvas.getBoundingClientRect();
  return { x: (e.clientX - r.left) * canvas.width / r.width, y: (e.clientY - r.top) * canvas.height / r.height };
}
canvas.addEventListener('pointerdown', e => {
  drawing = true; canvas.setPointerCapture(e.pointerId);
  history.push(snapshot()); paint(pt(e), null);
});
canvas.addEventListener('pointermove', e => { if (drawing) paint(pt(e), last); });
canvas.addEventListener('pointerup', () => { drawing = false; last = null; });
let last = null;
function paint(p, prev) {
  const size = +$('#brushSize').value;
  const layer = tool === 'brush' ? layers.add : layers.sub;
  const c = layer.getContext('2d');
  c.lineCap = c.lineJoin = 'round';
  c.lineWidth = size; c.strokeStyle = '#fff';
  c.beginPath();
  if (prev) { c.moveTo(prev.x, prev.y); c.lineTo(p.x, p.y); } else { c.moveTo(p.x, p.y); c.lineTo(p.x + .01, p.y); }
  c.stroke();
  last = p; draw();
}
function snapshot() {
  const c = document.createElement('canvas'); c.width = canvas.width; c.height = canvas.height;
  const x = c.getContext('2d'); x.drawImage(layers.add, 0, 0); return c;
}
function undo() {
  if (!history.length || !layers) return;
  const c = layers.add.getContext('2d'); c.clearRect(0, 0, canvas.width, canvas.height);
  c.drawImage(history.pop(), 0, 0); draw();
}

function applyMask() {
  const it = state.editing; if (!it) return;
  it.mode = $('#mode').value;
  const out = document.createElement('canvas'); out.width = canvas.width; out.height = canvas.height;
  const o = out.getContext('2d');
  o.drawImage(layers.auto, 0, 0);
  o.globalCompositeOperation = 'source-over'; o.drawImage(layers.add, 0, 0);
  o.globalCompositeOperation = 'destination-out'; o.drawImage(layers.sub, 0, 0);
  it.mask = out.toDataURL('image/png');
  it.maskUrl = it.mask;                 // preview my refined mask in the card
  $('#editor').hidden = true; render();
}
