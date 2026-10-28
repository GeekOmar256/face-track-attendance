export function escapeText(value) {
  return String(value ?? '').replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

export function asDate(value) {
  return value?.toDate ? value.toDate() : new Date(value || 0);
}

export function deviceOnline(data, now = Date.now()) {
  const age = now - asDate(data?.last_seen).getTime();
  return !!data?.last_seen && Number.isFinite(age) && age >= -10000 && age < 20000;
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
