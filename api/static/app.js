'use strict';
/*
 * LexProbe front end. Two parts:
 *   1. pure functions (sections, highlight segments, citation text, report text): no DOM, unit-tested with node
 *      (tests/js/test_app.js, run by tests/test_frontend.py);
 *   2. the page: renders results with DOM methods only (textContent / text nodes). Model output is NEVER put into
 *      innerHTML, so a quote cannot inject markup.
 */

const MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024;
const MAX_TEXT_CHARS = 15000;
const HUES = [215, 28, 165, 330, 265, 50, 190, 100];   // one hue per clause, stable by position in the response

// ── pure functions ──────────────────────────────────────────────────────────────────────────────
/** Headings that start a line: "4. Term and Termination. text", "3.1 Fees", "ARTICLE IV - TERM". Titles are 2-60 chars. */
function parseSections(text) {
  const found = [];
  const add = (re, numGroup, titleGroup) => {
    let m;
    while ((m = re.exec(text)) !== null) {
      const title = m[titleGroup].trim().replace(/[.:\-\s]+$/, '');
      if (title.length < 2 || title.length > 60) continue;
      found.push({ number: m[numGroup].replace(/\.$/, '').trim(), title, startIndex: m.index });
    }
  };
  add(/^[ \t]*(\d+(?:\.\d+)*)\.?[ \t]+([A-Z][A-Za-z0-9 ,;'&\/()\-]{1,58}?)\.[ \t]+\S/gm, 1, 2);
  add(/^[ \t]*(\d+(?:\.\d+)*)\.?[ \t]+([A-Z][A-Za-z0-9 ,;'&\/()\-]{1,58}?)\.?[ \t]*$/gm, 1, 2);
  add(/^[ \t]*((?:ARTICLE|SECTION|Article|Section)[ \t]+(?:[IVXLCDM]+|\d+))[ \t.:\-]+([A-Z][A-Za-z0-9 ,;'&\/()\-]{1,58})[ \t]*(?:$|\.)/gm, 1, 2);
  found.sort((a, b) => a.startIndex - b.startIndex);
  return found;
}

/** The nearest heading at or above a character position (null when there is none). */
function findSection(pos, sections) {
  if (pos === null || pos === undefined || pos < 0 || sections.length === 0) return null;
  let best = null;
  for (const sec of sections) {
    if (sec.startIndex <= pos) best = sec; else break;
  }
  return best;
}

/**
 * Split `text` at every span boundary. spans: [{id, start, end}]. Returns [{text, ids}] where ids lists every span that covers
 * the piece (overlaps are allowed). Invalid spans (non-integers, empty, outside the text) are ignored.
 */
function buildSegments(text, spans) {
  const valid = spans.filter(s => Number.isInteger(s.start) && Number.isInteger(s.end) && s.start >= 0 && s.end > s.start && s.end <= text.length);
  const cuts = new Set([0, text.length]);
  valid.forEach(s => { cuts.add(s.start); cuts.add(s.end); });
  const points = [...cuts].sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i], b = points[i + 1];
    if (b <= a) continue;
    out.push({ start: a, text: text.slice(a, b), ids: valid.filter(s => s.start <= a && s.end >= b).map(s => s.id) });
  }
  return out;
}

const fmt = n => Number(n).toLocaleString('en-US');

/** One-line citation a reviewer can paste into a memo. */
function citationText(clause, section, docName) {
  const where = section ? `, § ${section.number} (${section.title})` : '';
  return `“${clause.extracted_text}” — ${docName || 'Contract'}${where}, characters ${fmt(clause.start)}–${fmt(clause.end)}` +
    (clause.grounded === false ? ' [quote NOT found verbatim: verify manually]' : '');
}

/** Markdown report of the findings (flagged clauses with citations, then what was not found). */
function reportMarkdown(data, docName, sections) {
  const flagged = data.clauses.filter(c => c.detected);
  const lines = [
    `# LexProbe contract review: ${docName || 'Contract'}`, '',
    `${flagged.length} of ${data.total_checked} clause types flagged · ${data.chunks_analyzed} chunks analysed · model ${data.model_version} (${data.scoring}, threshold ${data.decision_threshold})`, '',
  ];
  flagged.forEach((c, i) => {
    const sec = findSection(c.start, sections);
    lines.push(`## ${i + 1}. ${c.clause_type}`, '');
    String(c.extracted_text || '').split('\n').forEach(l => lines.push(`> ${l}`));
    lines.push('', `- Location: ${sec ? `§ ${sec.number} (${sec.title}), ` : ''}characters ${fmt(c.start)}–${fmt(c.end)}, chunk ${c.chunk_index + 1} of ${data.chunks_analyzed}`,
      `- Verification: ${c.grounded ? 'quote found verbatim in the document' : 'NOT found verbatim: verify manually'}`, '');
  });
  const none = data.clauses.filter(c => !c.detected).map(c => c.clause_type);
  if (none.length) lines.push('## Not found in this document', '', none.map(n => `- ${n}`).join('\n'), '');
  lines.push('---', 'AI-assisted triage, not legal advice. A clause that is not flagged may still be present; read the source.');
  return lines.join('\n');
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { parseSections, findSection, buildSegments, citationText, reportMarkdown, fmt };
}

// ── the page ────────────────────────────────────────────────────────────────────────────────────
if (typeof document !== 'undefined') {
  const $ = id => document.getElementById(id);

  function el(tag, props, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k === 'style') String(v).split(';').filter(Boolean).forEach(d => { const [name, val] = d.split(':'); node.style.setProperty(name.trim(), val.trim()); });  // CSSOM, not an inline style attribute: allowed under a strict CSP
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined && v !== false) node.setAttribute(k, v === true ? '' : v);
    }
    children.flat().forEach(c => { if (c !== null && c !== undefined && c !== false) node.append(c); });
    return node;
  }

  let state = { data: null, docName: 'Pasted text', sections: [], active: null, timer: null };

  function setError(msg) {
    const box = $('error');
    box.textContent = msg || '';
    box.hidden = !msg;
  }

  function updateCounter() {
    const n = $('contract-text').value.length;
    const c = $('counter');
    c.textContent = `${fmt(n)} / ${fmt(MAX_TEXT_CHARS)} characters`;
    c.dataset.level = n > MAX_TEXT_CHARS ? 'over' : n > MAX_TEXT_CHARS * 0.9 ? 'near' : 'ok';
  }

  async function loadHealth() {
    const chip = $('model-chip');
    try {
      const h = await (await fetch('/health')).json();
      const clauses = await (await fetch('/clause-types')).json();
      chip.textContent = h.model_loaded ? `Model ready · ${h.scoring} · ${clauses.clause_types.length} clause types` : 'Model is still loading';
      chip.dataset.state = h.model_loaded ? 'ok' : 'warn';
    } catch (e) {
      chip.textContent = 'Server not reachable';
      chip.dataset.state = 'bad';
    }
  }

  function readTextFile(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = e => resolve(e.target.result);
      r.onerror = () => reject(new Error('Could not read the file.'));
      r.readAsText(file);
    });
  }

  async function onFileChosen(file) {
    if (!file) return;
    setError('');
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setError(`File is too large (${(file.size / 1048576).toFixed(2)} MB). The limit is 2 MB.`);
      $('file-input').value = '';
      return;
    }
    state.docName = file.name;
    $('file-name').textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    const ext = file.name.split('.').pop().toLowerCase();
    if (ext === 'txt') {
      $('contract-text').value = await readTextFile(file);
      $('contract-text').placeholder = 'Paste the contract text here…';
      if ($('contract-text').value.length > MAX_TEXT_CHARS) setError(`This text has ${fmt($('contract-text').value.length)} characters; the limit is ${fmt(MAX_TEXT_CHARS)} (about 5 pages). Shorten it or select an excerpt.`);
    } else {
      $('contract-text').value = '';
      $('contract-text').placeholder = `${file.name} selected. Its text is extracted on the server when you click Analyze.`;
    }
    updateCounter();
  }

  async function loadSample(path, title) {
    setError('');
    const r = await fetch(path);
    $('contract-text').value = await r.text();
    $('file-input').value = '';
    $('file-name').textContent = '';
    state.docName = title;
    updateCounter();
    $('contract-text').scrollTop = 0;
  }

  function startBusy() {
    $('analyze-btn').disabled = true;
    $('analyze-btn').textContent = 'Analysing…';
    $('progress').hidden = false;
    $('results').hidden = true;
    const t0 = Date.now();
    const tick = () => { $('elapsed').textContent = `${Math.round((Date.now() - t0) / 1000)} s`; };
    tick();
    state.timer = setInterval(tick, 1000);
  }

  function stopBusy() {
    clearInterval(state.timer);
    $('progress').hidden = true;
    $('analyze-btn').disabled = false;
    $('analyze-btn').textContent = 'Analyse contract';
  }

  async function analyze() {
    setError('');
    const file = $('file-input').files[0];
    const text = $('contract-text').value.trim();
    const ext = file ? file.name.split('.').pop().toLowerCase() : '';
    const serverExtract = file && ['pdf', 'docx'].includes(ext);
    if (!serverExtract && !text) { setError('Paste contract text or choose a file first.'); return; }
    if (!serverExtract && text.length > MAX_TEXT_CHARS) { setError(`The text has ${fmt(text.length)} characters; the limit is ${fmt(MAX_TEXT_CHARS)} (about 5 pages).`); return; }
    if (!serverExtract && text.length < 50) { setError('The text is too short to analyse (minimum 50 characters).'); return; }
    startBusy();
    try {
      let resp;
      if (serverExtract) {
        const form = new FormData();
        form.append('file', file);
        resp = await fetch('/upload', { method: 'POST', body: form });
      } else {
        resp = await fetch('/analyze-contract', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
      }
      if (!resp.ok) {
        let detail = `Server error ${resp.status}`;
        try { const j = await resp.json(); detail = typeof j.detail === 'string' ? j.detail : detail; } catch (e) { /* keep default */ }
        throw new Error(detail);
      }
      render(await resp.json());
    } catch (err) {
      setError(err.message || 'Something went wrong.');
    } finally {
      stopBusy();
    }
  }

  // ── rendering ──────────────────────────────────────────────────────────────────────────────
  const hue = i => HUES[i % HUES.length];

  function render(data) {
    state.data = data;
    state.sections = parseSections(data.contract_text || '');
    state.active = null;
    const flagged = data.clauses.filter(c => c.detected);
    const order = data.clauses.map((c, i) => ({ c, i })).sort((a, b) => (b.c.detected - a.c.detected) || (a.i - b.i));
    let n = 0;
    order.forEach(o => { o.n = o.c.detected ? ++n : null; });
    state.order = order;

    $('summary-count').textContent = `${flagged.length} of ${data.total_checked}`;
    $('summary-label').textContent = 'clause types flagged';
    $('summary-meta').textContent = `${data.chunks_analyzed} chunks analysed · ${data.model_version} · flag threshold ${data.decision_threshold}`;
    $('doc-name').textContent = state.docName;

    const chips = $('chips');
    chips.replaceChildren(...order.map(o => el('button', {
      type: 'button', class: 'chip', 'data-flagged': o.c.detected, 'aria-label': `${o.c.clause_type}: ${o.c.detected ? 'flagged, show citation' : 'not found'}`,
      style: `--h:${hue(o.i)}`, disabled: !o.c.detected, onclick: () => activate(o.i, true),
    }, el('span', { class: 'dot' }), o.c.clause_type)));

    const list = $('findings');
    list.replaceChildren(...order.map(o => o.c.detected ? flaggedCard(o) : null).filter(Boolean));
    const none = order.filter(o => !o.c.detected);
    $('not-found').hidden = none.length === 0;
    $('not-found-list').replaceChildren(...none.map(o => el('li', { text: o.c.clause_type })));

    renderDocument(data, order);
    $('results').hidden = false;
    $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function flaggedCard(o) {
    const c = o.c, data = state.data;
    const sec = findSection(c.start, state.sections);
    const long = (c.extracted_text || '').length > 420;
    const quote = el('blockquote', { class: 'quote', 'data-clamped': long }, el('p', { text: c.extracted_text || '' }));
    const verified = c.grounded
      ? el('span', { class: 'verify ok', title: 'The quoted text was found, character for character, in your document.' }, '✓ Verified verbatim in the source')
      : el('span', { class: 'verify warn', title: 'The model’s quote was not found exactly in the document. Treat it as unverified.' }, '⚠ Not found verbatim — verify manually');
    const where = [
      sec ? `§ ${sec.number} — ${sec.title}` : null,
      `characters ${fmt(c.start)}–${fmt(c.end)}`,
      `chunk ${c.chunk_index + 1} of ${data.chunks_analyzed}`,
    ].filter(Boolean).join(' · ');
    const copyBtn = el('button', { type: 'button', class: 'btn small', onclick: async ev => {
      try { await navigator.clipboard.writeText(citationText(c, sec, state.docName)); ev.target.textContent = 'Copied ✓'; }
      catch (e) { ev.target.textContent = 'Copy failed'; }
      setTimeout(() => { ev.target.textContent = 'Copy citation'; }, 1800);
    } }, 'Copy citation');
    const toggle = long ? el('button', { type: 'button', class: 'btn small ghost', 'aria-expanded': 'false', onclick: ev => {
      const open = quote.dataset.clamped === 'false';
      quote.dataset.clamped = open ? 'true' : 'false';
      ev.target.textContent = open ? 'Show full quote' : 'Show less';
      ev.target.setAttribute('aria-expanded', String(!open));
    } }, 'Show full quote') : null;
    const margin = c.confidence_margin !== null && c.confidence_margin !== undefined ? c.confidence_margin : null;
    const why = el('details', { class: 'why' },
      el('summary', { text: 'How was this decided?' }),
      el('p', { text: `The model scored each of the ${data.chunks_analyzed} chunks for this clause type. The highest score was ${margin !== null ? margin.toFixed(2) : 'n/a'}, ` +
        `above the flag threshold of ${data.decision_threshold}; ${c.chunks_flagged} chunk${c.chunks_flagged === 1 ? '' : 's'} cleared it. ` +
        'The score is a ranking signal, not a probability.' }));
    const card = el('article', { class: 'card finding', id: `finding-${o.i}`, style: `--h:${hue(o.i)}`, tabindex: '-1' },
      el('header', {}, el('span', { class: 'num', text: String(o.n) }), el('h3', { text: c.clause_type }), verified),
      quote,
      el('p', { class: 'where', text: where }),
      el('div', { class: 'actions' }, copyBtn, el('button', { type: 'button', class: 'btn small', onclick: () => activate(o.i, false) }, 'Show in document'), toggle),
      why);
    card.addEventListener('click', ev => { if (!ev.target.closest('button, summary, details')) activate(o.i, false, true); });
    return card;
  }

  function renderDocument(data, order) {
    const spans = order.filter(o => o.c.detected && Number.isInteger(o.c.start)).map(o => ({ id: o.i, start: o.c.start, end: o.c.end }));
    const segs = buildSegments(data.contract_text, spans);
    const numberOf = Object.fromEntries(order.filter(o => o.n).map(o => [o.i, o.n]));
    const started = new Set();
    const nodes = segs.map(s => {
      if (!s.ids.length) return document.createTextNode(s.text);
      const first = s.ids.filter(id => !started.has(id));
      first.forEach(id => started.add(id));
      const m = el('mark', { class: 'hl', 'data-ids': s.ids.join(' '), style: `--h:${hue(s.ids[s.ids.length - 1])}`, text: s.text });
      if (first.length) m.dataset.n = first.map(id => numberOf[id]).join(',');
      m.addEventListener('click', () => activate(s.ids[s.ids.length - 1], false, false, true));
      return m;
    });
    $('doc').replaceChildren(...nodes);
  }

  function activate(i, scrollCard, fromCard, fromDoc) {
    state.active = i;
    document.querySelectorAll('.finding').forEach(f => f.classList.toggle('active', f.id === `finding-${i}`));
    const marks = [...document.querySelectorAll('mark.hl')];
    marks.forEach(m => m.classList.toggle('active', m.dataset.ids.split(' ').includes(String(i))));
    const first = marks.find(m => m.classList.contains('active'));
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (first && !fromDoc) {
      const pane = $('doc-pane');
      if (getComputedStyle(pane).overflowY === 'auto') pane.scrollTo({ top: Math.max(0, first.offsetTop - pane.clientHeight / 3), behavior: reduce ? 'auto' : 'smooth' });
      else first.scrollIntoView({ block: 'center', behavior: reduce ? 'auto' : 'smooth' });
      first.classList.remove('flash'); void first.offsetWidth; first.classList.add('flash');
    }
    if (scrollCard || fromDoc) {
      const card = $(`finding-${i}`);
      if (card) { card.scrollIntoView({ block: 'center', behavior: reduce ? 'auto' : 'smooth' }); if (fromDoc) card.focus({ preventScroll: true }); }
    }
  }

  function downloadReport() {
    const blob = new Blob([JSON.stringify({ document: state.docName, ...state.data, contract_text: undefined }, null, 2)], { type: 'application/json' });
    const a = el('a', { href: URL.createObjectURL(blob), download: 'lexprobe-report.json' });
    document.body.append(a); a.click(); a.remove(); URL.revokeObjectURL(a.href);
  }

  async function copyReport(ev) {
    const btn = ev.currentTarget;
    try { await navigator.clipboard.writeText(reportMarkdown(state.data, state.docName, state.sections)); btn.textContent = 'Copied ✓'; }
    catch (e) { btn.textContent = 'Copy failed'; }
    setTimeout(() => { btn.textContent = 'Copy report (Markdown)'; }, 1800);
  }

  document.addEventListener('DOMContentLoaded', () => {
    $('analyze-btn').addEventListener('click', analyze);
    $('contract-text').addEventListener('input', () => { if (!$('file-input').files[0]) state.docName = 'Pasted text'; updateCounter(); });
    $('file-input').addEventListener('change', e => onFileChosen(e.target.files[0]));
    const drop = $('drop');
    ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
    drop.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) { const dt = new DataTransfer(); dt.items.add(f); $('file-input').files = dt.files; onFileChosen(f); } });
    $('sample-1').addEventListener('click', () => loadSample('/static/samples/sponsorship_agreement.txt', 'Sponsorship agreement (sample)'));
    $('sample-2').addEventListener('click', () => loadSample('/static/samples/ip_agreement.txt', 'Intellectual property agreement (sample)'));
    $('download-btn').addEventListener('click', downloadReport);
    $('copy-report-btn').addEventListener('click', copyReport);
    $('print-btn').addEventListener('click', () => window.print());
    $('again-btn').addEventListener('click', () => { $('input-card').scrollIntoView({ behavior: 'smooth' }); $('contract-text').focus(); });
    updateCounter();
    loadHealth();
  });
}
