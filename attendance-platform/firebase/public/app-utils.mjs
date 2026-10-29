export function escapeText(value) {
  return String(value ?? '').replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

export function asDate(value) {
  return value?.toDate ? value.toDate() : new Date(value || 0);
}

export function observeHeartbeat(previous, data, fromCache, receivedAt) {
  const stamp = asDate(data?.last_seen).getTime();
  if (fromCache || !data?.last_seen || !Number.isFinite(stamp)) return previous;
  if (previous?.stamp === stamp) return previous;
  // Two advancing server snapshots prove liveness without trusting the viewer's clock.
  return { stamp, at: previous && stamp > previous.stamp ? receivedAt : null };
}

export function deviceOnline(data, now = Date.now(), observation = null, monotonicNow = 0) {
  const stamp = asDate(data?.last_seen).getTime();
  const age = now - stamp;
  const observedAge = monotonicNow - observation?.at;
  const observedLive = observation?.stamp === stamp && observation.at !== null
    && Number.isFinite(observedAge) && observedAge >= 0 && observedAge < 20000;
  return !!data?.last_seen && Number.isFinite(age)
    && ((age >= -10000 && age < 20000) || observedLive);
}

export function uniqueAttendance(rows) {
  const seen = new Set();
  return [...rows].sort((first, second) => asDate(first.timestamp) - asDate(second.timestamp)).filter(row => {
    const key = `${row.session_id}/${row.user_id}`;
    if (!row.user_id || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function csvText(rows) {
  if (!rows.length) throw Error('There are no records to export.');
  const keys = [...new Set(rows.flatMap(row => Object.keys(row)))].filter(key => !['image', 'thumbnail'].includes(key));
  const cell = value => {
    let text = typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value ?? '');
    if (/^[\s\u0000-\u001f]*[=+@-]/u.test(text)) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  };
  return '\ufeff' + [keys.map(cell), ...rows.map(row => keys.map(key => cell(row[key])))].map(row => row.join(',')).join('\r\n');
}

export function filterRows(rows, search, keys) {
  const term = search.trim().toLocaleLowerCase();
  return rows.filter(row => keys.some(key => String(row[key] || '').toLocaleLowerCase().includes(term)));
}
