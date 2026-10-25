'use strict';
// Tests for the downloadable HTML report in api/static/app.js (run with: node tests/js/test_report.js).
const assert = require('assert');
const path = require('path');
const { parseSections, reportHtml, reportFileName, escapeHtml } = require(path.join(__dirname, '..', '..', 'api', 'static', 'app.js'));

let n = 0;
const test = (name, fn) => { fn(); n++; console.log('ok -', name); };

const TEXT = 'Intro paragraph.\n\n8. Miscellaneous. This Agreement shall be governed by Delaware law. Notices go to the address above.\n\n4. Term. Renews yearly unless ended.';
const gl = TEXT.indexOf('This Agreement shall be governed by Delaware law.');
const ar = TEXT.indexOf('Renews yearly');
const QUOTE = 'This Agreement shall be governed by Delaware law.';
const DATA = {
  total_checked: 3, chunks_analyzed: 4, model_version: 'llama-3.1-8b-cuad-lora-v6b', scoring: 'v6b', decision_threshold: -2, contract_text: TEXT,
  clauses: [
    { clause_type: 'Auto-Renewal', detected: true, extracted_text: 'Renews yearly', start: ar, end: ar + 13, chunk_index: 2, chunks_flagged: 1, confidence_margin: -1.2, grounded: true },
    { clause_type: 'Governing Law', detected: true, extracted_text: QUOTE, start: gl, end: gl + QUOTE.length, chunk_index: 1, chunks_flagged: 1, confidence_margin: 0.85, grounded: true },
    { clause_type: 'Non-Compete', detected: false },
  ],
};
const NOW = new Date('2026-10-23T12:30:00Z');

test('report file names are safe and dated', () => {
  assert.strictEqual(reportFileName('Sponsorship agreement (sample)', NOW), 'lexprobe-report-Sponsorship-agreement-sample-2026-10-23.html');
  assert.strictEqual(reportFileName('My Contract.pdf', NOW), 'lexprobe-report-My-Contract-2026-10-23.html');
  assert.strictEqual(reportFileName('../../etc/passwd<>:"|?*', NOW), 'lexprobe-report-etc-passwd-2026-10-23.html');
  assert.strictEqual(reportFileName('', NOW), 'lexprobe-report-contract-2026-10-23.html');
  assert.ok(reportFileName('x'.repeat(300), NOW).length < 100);
});

test('escapeHtml escapes all five dangerous characters', () => {
  assert.strictEqual(escapeHtml('<a href="x" onclick=\'y\'>&</a>'), '&lt;a href=&quot;x&quot; onclick=&#39;y&#39;&gt;&amp;&lt;/a&gt;');
});

test('report contains the summary, findings with citations, not-found list and disclaimer', () => {
  const html = reportHtml(DATA, 'Test contract.txt', parseSections(TEXT), NOW);
  assert.ok(html.startsWith('<!DOCTYPE html>'));
  assert.ok(html.includes('Summary: 2 of 3 clause types flagged'));
  assert.ok(html.includes('Flagged [1]') && html.includes('Flagged [2]') && html.includes('Not found'));
  assert.ok(html.includes('Verified verbatim in the source'));
  assert.ok(html.includes('characters ' + gl.toLocaleString('en-US')));
  assert.ok(html.includes('score 0.85 against a flag threshold of -2'));
  assert.ok(html.includes('Not found in this document') && html.includes('<li>Non-Compete</li>'));
  assert.ok(html.includes('2026-10-23 12:30 UTC') && html.includes('not legal advice'));
});

test('report marks each citation in the contract with a readable [n Clause] tag', () => {
  const html = reportHtml(DATA, 'Doc', parseSections(TEXT), NOW);
  assert.strictEqual((html.match(/<mark class="h"/g) || []).length, 2);
  assert.ok(html.includes('<span class="tag">[1 Auto-Renewal]</span>') && html.includes('<span class="tag">[2 Governing Law]</span>'));
  assert.ok(html.includes(QUOTE + '</mark>'));      // the highlighted text is exactly the cited quote
  assert.ok(html.includes('Renews yearly</mark>'));
});

test('report is safe: hostile quote, document name and contract text are shown as text, and a CSP forbids scripts', () => {
  const bad = '<img src=x onerror="alert(1)"><script>alert(2)</script>';
  const hostile = {
    ...DATA, contract_text: TEXT + '\n' + bad,
    clauses: [{ ...DATA.clauses[0], extracted_text: bad, start: TEXT.length + 1, end: TEXT.length + 1 + bad.length }, DATA.clauses[1], DATA.clauses[2]],
  };
  const html = reportHtml(hostile, '<script>x</script>.txt', parseSections(TEXT), NOW);
  const body = html.replace(/<style>[\s\S]*?<\/style>/, '');
  assert.ok(!/<img src=x/.test(body), 'no live <img> from data');
  assert.ok(!/<script/i.test(body), 'no live <script> from data');
  assert.ok(html.includes('&lt;img src=x onerror=&quot;alert(1)&quot;&gt;') && html.includes('&lt;script&gt;alert(2)&lt;/script&gt;'));
  assert.ok(html.includes('<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'));
  assert.ok(!html.includes('src="http') && !html.includes('href="http'), 'no external resources');
});

test('report handles a document with nothing flagged', () => {
  const none = { ...DATA, clauses: DATA.clauses.map(c => ({ clause_type: c.clause_type, detected: false })) };
  const html = reportHtml(none, 'Doc', [], NOW);
  assert.ok(html.includes('Summary: 0 of 3 clause types flagged') && html.includes('No clause type was flagged.'));
  assert.strictEqual((html.match(/<mark/g) || []).length, 0);
});

test('report copes with overlapping citations (one passage cited by two clauses)', () => {
  const inner = QUOTE.slice(5, 30);
  const both = { ...DATA, clauses: [DATA.clauses[1], { ...DATA.clauses[1], clause_type: 'Other', start: gl + 5, end: gl + 30, extracted_text: inner }, DATA.clauses[2]] };
  const html = reportHtml(both, 'Doc', [], NOW);
  assert.ok(html.includes('[1 Governing Law]') && html.includes('[2 Other]'));
  const plain = html.replace(/<span class="tag">[^<]*<\/span>/g, '').replace(/<[^>]+>/g, '');
  assert.ok(plain.includes(QUOTE), 'the cited passage is intact once tags are stripped');
});

test('report is responsive: the summary table scrolls inside its own container and text can wrap', () => {
  const html = reportHtml(DATA, 'Doc', parseSections(TEXT), NOW);
  assert.ok(html.includes('<div class="tw"><table>') && html.includes('</table></div>'));
  assert.ok(html.includes('.tw{overflow-x:auto}') && html.includes('@media(max-width:560px)'));
  assert.ok(!/th,td\{[^}]*overflow-wrap:anywhere/.test(html), 'table words are not broken mid-word');
});

console.log(`${n} report tests passed`);
