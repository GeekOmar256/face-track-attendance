import test from 'node:test';
import assert from 'node:assert/strict';
import { escapeText, deviceOnline, uniqueAttendance, csvText, filterRows } from '../public/app-utils.mjs';

test('device connectivity rejects stale and implausibly future heartbeats', () => {
  const now = Date.now();
  assert.equal(deviceOnline({ last_seen: new Date(now - 1000).toISOString() }, now), true);
  assert.equal(deviceOnline({ last_seen: new Date(now - 21000).toISOString() }, now), false);
  assert.equal(deviceOnline({ last_seen: new Date(now + 20000).toISOString() }, now), false);
  assert.equal(deviceOnline({ last_seen: 'invalid' }, now), false);
  assert.equal(deviceOnline(null, now), false);
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
