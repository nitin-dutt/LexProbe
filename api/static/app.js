'use strict';
/*
 * LexProbe front end. Two parts:
 *   1. pure functions (sections, highlight segments, citation text, Markdown report, downloadable HTML report): no DOM,
 *      unit-tested with node (tests/js/test_app.js, run by tests/test_frontend.py);
 *   2. the page: three screens (start, analysing, results). Model output is NEVER put into innerHTML: the page uses
 *      DOM methods and text nodes, and the downloadable report escapes every value and carries a CSP that forbids scripts.
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
 * Split `text` at every span boundary. spans: [{id, start, end}]. Returns [{start, text, ids}] where ids lists every span that
 * covers the piece (overlaps are allowed). Invalid spans (non-integers, empty, outside the text) are ignored.
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
const escapeHtml = s => String(s).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));

/** One-line citation a reviewer can paste into a memo. */
function citationText(clause, section, docName) {
  const where = section ? `, § ${section.number} (${section.title})` : '';
  return `“${clause.extracted_text}” — ${docName || 'Contract'}${where}, characters ${fmt(clause.start)}–${fmt(clause.end)}` +
    (clause.grounded === false ? ' [quote NOT found verbatim: verify manually]' : '');
}

/** Markdown report of the findings (flagged clauses with citations, then what was not found). */
/** A file name that is safe on every system: lexprobe-report-<document>-<YYYY-MM-DD>.html */
function reportFileName(docName, date) {
  const base = String(docName || 'contract').replace(/\.[A-Za-z0-9]{1,5}$/, '').replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^[-.]+|[-.]+$/g, '').slice(0, 60) || 'contract';
  return `lexprobe-report-${base}-${date.toISOString().slice(0, 10)}.html`;
}

const REPORT_CSS = `
:root{color-scheme:light}*{box-sizing:border-box}
body{margin:0;background:#fff;color:#14172b;font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;-webkit-print-color-adjust:exact;print-color-adjust:exact}
main{max-width:900px;margin:0 auto;padding:32px 24px 48px}
h1{margin:0 0 4px;font-size:1.7rem}h2{margin:30px 0 10px;font-size:1.2rem;border-bottom:2px solid #d9dcec;padding-bottom:4px}h3{margin:0;font-size:1.05rem}
.doc{font-size:1.1rem;font-weight:600;margin:0}.meta,.muted{color:#566089;font-size:.88rem}
.disclaimer{margin:18px 0;padding:10px 14px;border:1px solid #e0b25c;background:#fff6e0;border-radius:8px}
.tw{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:.92rem}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #e3e6f3;vertical-align:top}td:nth-child(2),td:nth-child(4){white-space:nowrap}th{background:#f2f4fb}
.finding{margin:14px 0;padding:12px 14px;border:1px solid #cfd4ea;border-left:6px solid hsl(var(--h) 70% 45%);border-radius:8px;break-inside:avoid}
.finding header{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.num{display:inline-block;min-width:24px;padding:1px 6px;text-align:center;border-radius:12px;background:hsl(var(--h) 70% 40%);color:#fff;font-weight:700;font-size:.85rem}
.ok{color:#0a6b46;font-weight:600;font-size:.85rem}.warn{color:#9a5200;font-weight:700;font-size:.85rem}
blockquote{margin:10px 0;padding:8px 12px;background:#f5f6fc;border-left:3px solid hsl(var(--h) 70% 45%);font-family:Georgia,"Times New Roman",serif;font-size:1rem;white-space:pre-wrap;overflow-wrap:anywhere}
.contract{white-space:pre-wrap;overflow-wrap:anywhere;font-family:Georgia,"Times New Roman",serif;font-size:.95rem;line-height:1.65;border:1px solid #cfd4ea;border-radius:8px;padding:14px 16px}
mark.h{color:inherit;background:hsl(var(--h) 90% 82%);text-decoration:underline;text-decoration-thickness:2px;text-underline-offset:2px;text-decoration-color:hsl(var(--h) 70% 35%)}
.tag{display:inline-block;margin-right:4px;padding:0 5px;border:1px solid #14172b;border-radius:4px;background:#fff;color:#14172b;font:700 .72rem system-ui,sans-serif;vertical-align:1px;text-decoration:none}
footer{margin-top:30px;font-size:.82rem;color:#566089}
@media(max-width:560px){main{padding:20px 14px 36px}th,td{padding:5px 6px;font-size:.82rem}h1{font-size:1.4rem}}
@media print{main{padding:0}h2{break-after:avoid}}`;

/**
 * A self-contained HTML report: summary table, every finding with its citation, and the full contract with each cited passage
 * highlighted and tagged "[n Clause]" (readable in black and white). Every value is escaped; the page carries a CSP that forbids
 * scripts and network access, so even a hostile quote cannot execute anything.
 */
function reportHtml(data, docName, sections, now) {
  const e = escapeHtml;
  const flagged = data.clauses.map((c, i) => ({ c, i })).filter(o => o.c.detected);
  const number = {};
  flagged.forEach((o, k) => { number[o.i] = k + 1; });
  const hueOf = i => HUES[i % HUES.length];
  const when = `${now.toISOString().slice(0, 10)} ${now.toISOString().slice(11, 16)} UTC`;
  const rows = data.clauses.map((c, i) => {
    const sec = c.detected ? findSection(c.start, sections) : null;
    const where = c.detected ? `${sec ? `§ ${e(sec.number)} — ${e(sec.title)}, ` : ''}characters ${fmt(c.start)}–${fmt(c.end)}` : '—';
    const ver = !c.detected ? '—' : c.grounded ? '<span class="ok">✓ verbatim</span>' : '<span class="warn">⚠ not verbatim</span>';
    return `<tr><td>${e(c.clause_type)}</td><td>${c.detected ? `Flagged [${number[i]}]` : 'Not found'}</td><td>${where}</td><td>${ver}</td></tr>`;
  }).join('');
  const findings = flagged.map(o => {
    const c = o.c, sec = findSection(c.start, sections);
    const score = c.confidence_margin === null || c.confidence_margin === undefined ? 'n/a' : Number(c.confidence_margin).toFixed(2);
    return `<article class="finding" style="--h:${hueOf(o.i)}"><header><span class="num">${number[o.i]}</span><h3>${e(c.clause_type)}</h3>` +
      (c.grounded ? '<span class="ok">✓ Verified verbatim in the source</span>' : '<span class="warn">⚠ Not found verbatim — verify manually</span>') +
      `</header><blockquote>${e(c.extracted_text || '')}</blockquote>` +
      `<p class="muted">${sec ? `§ ${e(sec.number)} — ${e(sec.title)} · ` : ''}characters ${fmt(c.start)}–${fmt(c.end)} · chunk ${Number(c.chunk_index) + 1} of ${fmt(data.chunks_analyzed)} · ` +
      `score ${e(score)} against a flag threshold of ${e(data.decision_threshold)} (a ranking signal, not a probability)</p></article>`;
  }).join('');
  const none = data.clauses.filter(c => !c.detected).map(c => `<li>${e(c.clause_type)}</li>`).join('');
  const spans = flagged.filter(o => Number.isInteger(o.c.start)).map(o => ({ id: o.i, start: o.c.start, end: o.c.end }));
  const started = new Set();
  const marked = buildSegments(data.contract_text || '', spans).map(s => {
    if (!s.ids.length) return e(s.text);
    const tags = s.ids.filter(id => !started.has(id)).map(id => { started.add(id); return `<span class="tag">[${number[id]} ${e(data.clauses[id].clause_type)}]</span>`; }).join('');
    return `<mark class="h" style="--h:${hueOf(s.ids[s.ids.length - 1])}">${tags}${e(s.text)}</mark>`;
  }).join('');
  return `<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">` +
    `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"><title>LexProbe review: ${e(docName || 'Contract')}</title><style>${REPORT_CSS}</style></head><body><main>` +
    `<header><h1>LexProbe contract review</h1><p class="doc">${e(docName || 'Contract')}</p>` +
    `<p class="meta">Generated ${e(when)} · model ${e(data.model_version)} (${e(data.scoring)}) · ${fmt(data.chunks_analyzed)} chunks analysed · flag threshold ${e(data.decision_threshold)}</p></header>` +
    `<div class="disclaimer"><strong>AI-assisted triage, not legal advice.</strong> A clause that is not flagged may still be present, and a flagged clause may be a false alarm. Read every finding in context. This file contains the full text of the contract.</div>` +
    `<h2>Summary: ${flagged.length} of ${data.total_checked} clause types flagged</h2><div class="tw"><table><thead><tr><th>Clause type</th><th>Result</th><th>Location</th><th>Quote check</th></tr></thead><tbody>${rows}</tbody></table></div>` +
    `<h2>Findings and citations</h2>${findings || '<p class="muted">No clause type was flagged.</p>'}` +
    (none ? `<h2>Not found in this document</h2><ul>${none}</ul><p class="muted">“Not found” means the model did not flag the clause; it does not prove the clause is absent.</p>` : '') +
    `<h2>Marked contract</h2><p class="muted">Each highlighted passage carries the number and clause type of its finding.</p><div class="contract">${marked}</div>` +
    `<footer>Quotes are checked character by character against the text supplied. Produced by LexProbe (Llama-3.1-8B-Instruct with a QLoRA adapter, trained on CUAD).</footer></main></body></html>`;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { parseSections, findSection, buildSegments, citationText, reportHtml, reportFileName, escapeHtml, fmt };
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

  let state = { data: null, docName: 'Pasted text', sections: [], active: null, timer: null, screen: 'input', fromFile: false };

  // ── screens ──────────────────────────────────────────────────────────────────────────────
  function showScreen(name) {
    state.screen = name;
    $('screen-input').hidden = name !== 'input';
    $('screen-busy').hidden = name !== 'busy';
    $('screen-results').hidden = name !== 'results';
    $('new-btn').hidden = name !== 'results';
    document.title = name === 'results' ? `${state.docName} · LexProbe` : 'LexProbe · Contract risk triage with verified citations';
    const back = $('back-link');
    back.hidden = !(name === 'input' && state.data);
    if (!back.hidden) back.textContent = `← Back to the results for ${state.docName}`;
    window.scrollTo({ top: 0, behavior: 'instant' });
    const heading = { input: 'h-input', busy: 'h-busy', results: 'h-results' }[name];
    $(heading).setAttribute('tabindex', '-1');
    $(heading).focus({ preventScroll: true });
  }

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

  // ── input: file, paste, samples, drag and drop ───────────────────────────────────────────
  function readTextFile(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = e => resolve(e.target.result);
      r.onerror = () => reject(new Error('Could not read the file.'));
      r.readAsText(file);
    });
  }

  function clearFile() {
    $('file-input').value = '';
    $('file-chip').hidden = true;
    $('drop').hidden = false;
    if (state.fromFile) $('contract-text').value = '';
    state.fromFile = false;
    $('contract-text').disabled = false;
    $('contract-text').placeholder = 'Paste the contract text here…';
    state.docName = 'Pasted text';
    updateCounter();
  }

  async function onFileChosen(file) {
    if (!file) return;
    setError('');
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['txt', 'pdf', 'docx'].includes(ext)) { setError('Please choose a .pdf, .docx or .txt file.'); $('file-input').value = ''; return; }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setError(`File is too large (${(file.size / 1048576).toFixed(2)} MB). The limit is 2 MB.`);
      $('file-input').value = '';
      return;
    }
    state.docName = file.name;
    $('file-name').textContent = file.name;
    $('file-size').textContent = ` · ${(file.size / 1024).toFixed(1)} KB`;
    $('file-chip').hidden = false;
    $('drop').hidden = true;
    state.fromFile = true;
    if (ext === 'txt') {
      $('contract-text').disabled = false;
      $('contract-text').value = await readTextFile(file);
      if ($('contract-text').value.length > MAX_TEXT_CHARS) setError(`This text has ${fmt($('contract-text').value.length)} characters; the limit is ${fmt(MAX_TEXT_CHARS)} (about 5 pages). Shorten it or select an excerpt.`);
    } else {
      $('contract-text').value = '';
      $('contract-text').disabled = true;
      $('contract-text').placeholder = `${file.name} will be read on the server when you click Analyse.`;
    }
    updateCounter();
  }

  function adoptFile(file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    $('file-input').files = dt.files;
    return onFileChosen(file);
  }

  async function loadSample(path, title) {
    setError('');
    clearFile();
    const r = await fetch(path);
    $('contract-text').value = await r.text();
    state.docName = title;
    updateCounter();
    $('contract-text').scrollTop = 0;
  }

  // ── analysis ─────────────────────────────────────────────────────────────────────────────
  function startBusy() {
    $('busy-name').textContent = state.docName;
    showScreen('busy');
    const t0 = Date.now();
    const tick = () => { $('elapsed').textContent = `${Math.round((Date.now() - t0) / 1000)} s`; };
    tick();
    state.timer = setInterval(tick, 1000);
  }

  function stopBusy() { clearInterval(state.timer); }

  async function analyze() {
    setError('');
    const file = $('file-input').files[0];
    const text = $('contract-text').value.trim();
    const ext = file ? file.name.split('.').pop().toLowerCase() : '';
    const serverExtract = file && ['pdf', 'docx'].includes(ext);
    if (!serverExtract && !text) { setError('Choose a file or paste the contract text first.'); return; }
    if (!serverExtract && text.length > MAX_TEXT_CHARS) { setError(`The text has ${fmt(text.length)} characters; the limit is ${fmt(MAX_TEXT_CHARS)} (about 5 pages).`); return; }
    if (!serverExtract && text.length < 50) { setError('The text is too short to analyse (minimum 50 characters).'); return; }
    $('analyze-btn').disabled = true;
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
      const data = await resp.json();
      stopBusy();
      render(data);
      history.pushState({ s: 'results' }, '');
    } catch (err) {
      stopBusy();
      showScreen('input');
      setError(err.message || 'Something went wrong.');
    } finally {
      $('analyze-btn').disabled = false;
    }
  }

  // ── results ──────────────────────────────────────────────────────────────────────────────
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

    $('results-title').textContent = state.docName;
    $('summary-count').textContent = `${flagged.length} of ${data.total_checked}`;
    $('summary-label').textContent = 'clause types flagged';
    $('summary-meta').textContent = `${data.chunks_analyzed} chunks analysed · ${data.model_version} · flag threshold ${data.decision_threshold}`;
    $('doc-name').textContent = state.docName;

    $('chips').replaceChildren(...order.map(o => el('button', {
      type: 'button', class: 'chip', 'data-flagged': o.c.detected, 'aria-label': `${o.c.clause_type}: ${o.c.detected ? 'flagged, show citation' : 'not found'}`,
      style: `--h:${hue(o.i)}`, disabled: !o.c.detected, onclick: () => activate(o.i, true),
    }, el('span', { class: 'dot' }), o.c.clause_type)));

    $('findings').replaceChildren(...order.map(o => o.c.detected ? flaggedCard(o) : null).filter(Boolean));
    const none = order.filter(o => !o.c.detected);
    $('not-found').hidden = none.length === 0;
    $('not-found-list').replaceChildren(...none.map(o => el('li', { text: o.c.clause_type })));

    renderDocument(data, order);
    showScreen('results');
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
    $('doc-pane').scrollTop = 0;
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

  // ── exports ──────────────────────────────────────────────────────────────────────────────
  function saveBlob(blob, name) {
    const a = el('a', { href: URL.createObjectURL(blob), download: name });
    document.body.append(a); a.click(); a.remove(); URL.revokeObjectURL(a.href);
  }

  function downloadReport() {
    const now = new Date();
    const html = reportHtml(state.data, state.docName, state.sections, now);
    saveBlob(new Blob([html], { type: 'text/html;charset=utf-8' }), reportFileName(state.docName, now));
  }

  // The browser's print dialog ("Save as PDF") prints the results page; the print CSS includes the marked contract.
  function downloadPdf() {
    const old = document.title;
    document.title = reportFileName(state.docName, new Date()).replace(/\.html$/, '');   // becomes the suggested PDF file name
    window.print();
    document.title = old;
  }

  function analyseAnother() {
    if (history.state && history.state.s === 'results') history.back();   // popstate shows the start screen
    else showScreen('input');
  }

  document.addEventListener('DOMContentLoaded', () => {
    history.replaceState({ s: 'input' }, '');
    $('analyze-btn').addEventListener('click', analyze);
    $('new-btn').addEventListener('click', analyseAnother);
    $('back-link').addEventListener('click', e => { e.preventDefault(); if (state.data) { showScreen('results'); history.pushState({ s: 'results' }, ''); } });
    window.addEventListener('popstate', e => {
      if (e.state && e.state.s === 'results' && state.data) showScreen('results');
      else if (state.screen !== 'busy') showScreen('input');
    });
    $('contract-text').addEventListener('input', () => {
      if ($('file-input').files[0]) { $('file-input').value = ''; $('file-chip').hidden = true; $('drop').hidden = false; state.fromFile = false; }
      state.docName = 'Pasted text';
      updateCounter();
    });
    $('file-input').addEventListener('change', e => onFileChosen(e.target.files[0]));
    $('file-remove').addEventListener('click', clearFile);
    const drop = $('drop');
    ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, () => drop.classList.remove('over')));
    // dropping a file anywhere on the start screen works
    let depth = 0;
    const hasFiles = e => e.dataTransfer && [...(e.dataTransfer.types || [])].includes('Files');
    window.addEventListener('dragenter', e => { if (state.screen === 'input' && hasFiles(e)) { depth++; $('drop-overlay').hidden = false; } });
    window.addEventListener('dragleave', e => { if (hasFiles(e)) { depth = Math.max(0, depth - 1); if (!depth) $('drop-overlay').hidden = true; } });
    window.addEventListener('dragover', e => { if (hasFiles(e)) e.preventDefault(); });
    window.addEventListener('drop', e => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      $('drop-overlay').hidden = true;
      if (state.screen === 'input' && e.dataTransfer.files[0]) adoptFile(e.dataTransfer.files[0]);
    });
    $('sample-1').addEventListener('click', () => loadSample('/static/samples/sponsorship_agreement.txt', 'Sponsorship agreement (sample)'));
    $('sample-2').addEventListener('click', () => loadSample('/static/samples/ip_agreement.txt', 'Intellectual property agreement (sample)'));
    $('report-btn').addEventListener('click', downloadReport);
    $('pdf-btn').addEventListener('click', downloadPdf);
    updateCounter();
    loadHealth();
  });
}
