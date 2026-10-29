import test from 'node:test';
import assert from 'node:assert/strict';
import { escapeText, deviceOnline, observeHeartbeat, uniqueAttendance, csvText, filterRows } from '../public/app-utils.mjs';

test('device connectivity rejects stale and implausibly future heartbeats', () => {
  const now = Date.now();
  assert.equal(deviceOnline({ last_seen: new Date(now - 1000).toISOString() }, now), true);
  assert.equal(deviceOnline({ last_seen: new Date(now - 21000).toISOString() }, now), false);
  assert.equal(deviceOnline({ last_seen: new Date(now + 20000).toISOString() }, now), false);
  assert.equal(deviceOnline({ last_seen: 'invalid' }, now), false);
  assert.equal(deviceOnline(null, now), false);
});

test('advancing server heartbeats remain online despite a wrong viewer clock and expire', () => {
  const first = { last_seen: '2026-10-29T18:00:00Z' };
  const next = { last_seen: '2026-10-29T18:00:02Z' };
  const wrongClock = Date.parse('2026-10-30T22:00:00Z');
  let observed = observeHeartbeat(null, first, false, 100);
  assert.equal(deviceOnline(first, wrongClock, observed, 100), false);
  observed = observeHeartbeat(observed, next, false, 2100);
  assert.equal(deviceOnline(next, wrongClock, observed, 2200), true);
  assert.equal(deviceOnline(next, Date.parse('2026-10-20'), observed, 2200), true);
  assert.equal(deviceOnline(next, wrongClock, observed, 22100), false);
  assert.equal(observeHeartbeat(observed, next, false, 25000), observed);
});

test('cached, malformed and backwards heartbeats cannot renew observed liveness', () => {
  const first = { last_seen: '2026-10-29T18:00:00Z' };
  const next = { last_seen: '2026-10-29T18:00:02Z' };
  const observed = observeHeartbeat(null, first, false, 100);
  assert.equal(observeHeartbeat(observed, next, true, 200), observed);
  assert.equal(observeHeartbeat(observed, { last_seen: 'invalid' }, false, 200), observed);
  assert.equal(observeHeartbeat(observed, { last_seen: '2020-01-01' }, false, 200).at, null);
  assert.equal(deviceOnline(first, Date.parse('2026-10-30'), observed, 200), false);
});

test('attendance keeps the earliest record per person within each session', () => {
  const rows = [
    { session_id: 'first', user_id: '1', timestamp: '2026-10-28T10:02:00Z' },
    { session_id: 'first', user_id: '1', timestamp: '2026-10-28T10:01:00Z' },
    { session_id: 'second', user_id: '1', timestamp: '2026-10-28T11:01:00Z' },
    { session_id: 'first', user_id: null, timestamp: '2026-10-28T10:00:00Z' }
  ];
  const selected = uniqueAttendance(rows);
  assert.equal(selected.length, 2);
  assert.equal(selected[0].timestamp, '2026-10-28T10:01:00Z');
  assert.equal(rows.length, 4);
});

test('CSV excludes photos, includes later columns and escapes formula injection', () => {
  const output = csvText([{ name: '\t=IMPORTXML("x")', image: 'private', thumbnail: 'private' }, { name: 'A,B', class_name: 'Class 1' }]);
  assert.ok(output.includes("'\t=IMPORTXML"));
  assert.ok(output.includes('"class_name"'));
  assert.ok(output.includes('"A,B"'));
  assert.ok(!output.includes('private'));
  assert.throws(() => csvText([]), /no records/);
});

test('student rendering and filtering handle untrusted profile values', () => {
  assert.equal(escapeText('<img src=x>'), '&lt;img src=x&gt;');
  assert.equal(filterRows([{ name: 'Mona', id: '101', class_name: 'CE492' }], ' ce492 ', ['name', 'id', 'class_name']).length, 1);
});
