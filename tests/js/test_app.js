'use strict';
// Pure-function tests for api/static/app.js (run with: node tests/js/test_app.js). Exits non-zero on the first failure.
const assert = require('assert');
const path = require('path');
const { parseSections, findSection, buildSegments, citationText } = require(path.join(__dirname, '..', '..', 'api', 'static', 'app.js'));

let n = 0;
const test = (name, fn) => { fn(); n++; console.log('ok -', name); };

const SPONSOR = [
  '1', '', 'CONFIDENTIAL TREATMENT HAS BEEN REQUESTED FOR CERTAIN REDACTED PROVISIONS OF THIS AGREEMENT. THE REDACTED PROVISIONS ARE IDENTIFIED BY THREE ASTERISKS.', '',
  '1. Form of Sponsorship. During the term of this Agreement, Tickets shall pay.', '',
  '4. Term and Termination. This Agreement shall commence on the Effective Date.', '',
  '8. Miscellaneous. This Agreement may not be assigned by Tickets without consent.',
].join('\n');

const IP = [
  '1.1 Certain Definitions.', '', 'As used in this Agreement:', '',
  '2. GRANT OF LICENSE', '', '2.1 MS Grant.', '', 'MS hereby grants a license.', '',
  '6.1 Governing Law; Jurisdiction; Dispute Resolution.', '', '(a) This Agreement shall be construed under New York law.',
].join('\n');

test('a page number followed by a long all-caps notice is not a section (the demo bug)', () => {
  const secs = parseSections(SPONSOR);
  assert.deepStrictEqual(secs.map(s => s.number), ['1', '4', '8']);
  assert.ok(!secs.some(s => /CONFIDENTIAL/.test(s.title)));
});

test('numbered headings with text on the same line, alone on a line, and with a final period', () => {
  assert.deepStrictEqual(parseSections(SPONSOR).map(s => s.title), ['Form of Sponsorship', 'Term and Termination', 'Miscellaneous']);
  assert.deepStrictEqual(parseSections(IP).map(s => `${s.number}:${s.title}`),
    ['1.1:Certain Definitions', '2:GRANT OF LICENSE', '2.1:MS Grant', '6.1:Governing Law; Jurisdiction; Dispute Resolution']);
});

test('article headings are recognised', () => {
  const secs = parseSections('ARTICLE IV - TERM\n\nText.\n\nSECTION 2: FEES\n\nMore text.');
  assert.deepStrictEqual(secs.map(s => s.number), ['ARTICLE IV', 'SECTION 2']);
});

test('findSection returns the nearest heading above a position, or null', () => {
  const secs = parseSections(SPONSOR);
  assert.strictEqual(findSection(SPONSOR.indexOf('This Agreement may not be assigned'), secs).title, 'Miscellaneous');
  assert.strictEqual(findSection(SPONSOR.indexOf('shall commence'), secs).title, 'Term and Termination');
  assert.strictEqual(findSection(0, secs), null);
  assert.strictEqual(findSection(null, secs), null);
  assert.strictEqual(findSection(10, []), null);
});

test('buildSegments splits at span boundaries and keeps the text intact', () => {
  const text = 'aaa BBB ccc DDD eee';
  const segs = buildSegments(text, [{ id: 1, start: 4, end: 7 }, { id: 2, start: 12, end: 15 }]);
  assert.strictEqual(segs.map(s => s.text).join(''), text);
  assert.deepStrictEqual(segs.map(s => s.ids), [[], [1], [], [2], []]);
});

test('buildSegments handles overlapping spans (both ids on the shared part)', () => {
  const segs = buildSegments('0123456789', [{ id: 1, start: 2, end: 6 }, { id: 2, start: 4, end: 8 }]);
  assert.strictEqual(segs.map(s => s.text).join(''), '0123456789');
  assert.deepStrictEqual(segs.find(s => s.text === '45').ids, [1, 2]);
  assert.deepStrictEqual(segs.find(s => s.text === '23').ids, [1]);
  assert.deepStrictEqual(segs.find(s => s.text === '67').ids, [2]);
});

test('buildSegments ignores invalid spans instead of breaking the page', () => {
  const text = 'abcdef';
  for (const bad of [{ start: null, end: 3 }, { start: 3, end: 3 }, { start: 4, end: 2 }, { start: -1, end: 2 }, { start: 2, end: 99 }, { start: 1.5, end: 3 }]) {
    const segs = buildSegments(text, [{ id: 1, ...bad }]);
    assert.strictEqual(segs.length, 1);
    assert.deepStrictEqual(segs[0].ids, []);
    assert.strictEqual(segs[0].text, text);
  }
});

test('buildSegments never turns text into markup: segments are plain strings', () => {
  const text = 'x <img src=x onerror=alert(1)> y';
  const segs = buildSegments(text, [{ id: 1, start: 2, end: 30 }]);
  assert.strictEqual(segs.map(s => s.text).join(''), text);
});

test('citationText gives quote, document, section and exact character range', () => {
  const clause = { extracted_text: 'This Agreement may not be assigned.', start: 6592, end: 6884, grounded: true };
  const c = citationText(clause, { number: '8', title: 'Miscellaneous' }, 'Sponsorship agreement');
  assert.strictEqual(c, '“This Agreement may not be assigned.” — Sponsorship agreement, § 8 (Miscellaneous), characters 6,592–6,884');
  assert.ok(citationText(clause, null, '').startsWith('“'));
  assert.ok(citationText({ ...clause, grounded: false }, null, 'Doc').includes('NOT found verbatim'));
});

console.log(`${n} frontend tests passed`);
