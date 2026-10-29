import { initializeApp } from 'https://www.gstatic.com/firebasejs/10.12.5/firebase-app.js';
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, updateProfile, onAuthStateChanged, signOut } from 'https://www.gstatic.com/firebasejs/10.12.5/firebase-auth.js';
import { getFirestore, collection, doc, onSnapshot, query, orderBy, limit, where, addDoc, setDoc, getDoc, getDocs, serverTimestamp, startAfter } from 'https://www.gstatic.com/firebasejs/10.12.5/firebase-firestore.js';
import { escapeText, asDate, deviceOnline, observeHeartbeat, uniqueAttendance, csvText, filterRows } from './app-utils.mjs';

const app = initializeApp(window.FIREBASE_CONFIG);
const auth = getAuth(app);
const database = getFirestore(app);
const device = window.DEVICE_ID;
const element = id => document.getElementById(id);
const collectionFor = name => collection(database, 'face_track_devices', device, name);
const labels = { haar: 'Haar Cascade', yunet: 'YuNet', dlib: 'dlib', lbph: 'LBPH', sface: 'SFace' };
const defaults = { detector: 'haar', recognizer: 'lbph', gate_enabled: true, camera: 'auto', camera_index: 0, rotation: 0, interval_ms: 100, haar_scale: 1.1, haar_neighbors: 5, yunet_score: .7, lbph_threshold: 70, sface_threshold: .363, dlib_threshold: .6 };
let users = [], sessions = [], attendance = [], subscriptions = [], deviceData = null, settings = { ...defaults };
let selected = '', followCurrent = true, settingsDirty = false, attendanceSubscription = null, accountSubscription = null;
let generation = 0, editing = false, signup = false, registration = null, historyCursor = null, hasEarlier = false;
let photoUrls = [], historyExtra = [], loadingHistory = false;
let studentSaving = false;
let heartbeat = null;
const pending = new Map();
const sending = new Set();
const pageTitles = { home: 'Classroom overview', students: 'Your students', history: 'Session history', models: 'Face models', device: 'Your device' };
const icons = {
  home: '<path d="m3 10 9-7 9 7v10H15v-7H9v7H3z"/>',
  users: '<circle cx="9" cy="8" r="3"/><path d="M3 21v-3a6 6 0 0 1 12 0v3M16 5a3 3 0 0 1 0 6m2 4a5 5 0 0 1 3 5"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  model: '<rect x="5" y="5" width="14" height="14" rx="3"/><path d="M9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m-4 6h4m14-6h4m-4 6h4M9 9h6v6H9z"/>',
  device: '<rect x="4" y="3" width="16" height="18" rx="3"/><path d="M8 7h8M8 11h4m-4 6h1m4 0h3"/>',
  check: '<path d="m5 12 4 4L19 6"/>'
};
for (const holder of document.querySelectorAll('[data-icon]')) holder.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${icons[holder.dataset.icon]}</svg>`;

const advanced = [
  ['camera', 'Camera source', [['auto', 'Automatic'], ['pi', 'Pi camera'], ['usb', 'USB / webcam']]],
  ['rotation', 'Camera rotation', [[0, '0°'], [90, '90°'], [180, '180°'], [270, '270°']]],
  ['camera_index', 'Camera index', 0, 20, 1], ['interval_ms', 'Frame interval (ms)', 0, 10000, 10],
  ['haar_scale', 'Haar scale factor', 1.01, 2, .01], ['haar_neighbors', 'Haar neighbors', 1, 30, 1],
  ['yunet_score', 'YuNet confidence', .01, 1, .01], ['lbph_threshold', 'LBPH maximum distance', 0, 1000, 'any'],
  ['sface_threshold', 'SFace minimum similarity', -1, 1, .001], ['dlib_threshold', 'dlib maximum distance', .01, 2, .01]
];
element('advancedFields').innerHTML = advanced.map(([key, title, low, high, step]) => `<div><label for="setting-${key}">${title}</label>${Array.isArray(low) ? `<select id="setting-${key}" name="${key}">${low.map(([value, label]) => `<option value="${value}">${label}</option>`).join('')}</select>` : `<input id="setting-${key}" name="${key}" type="number" min="${low}" max="${high}" step="${step}" required>`}</div>`).join('');

function notify(text, failure = false) {
  element('message').textContent = text;
  element('message').classList.toggle('error', failure);
  element('message').classList.toggle('hidden', !text);
}

function friendly(error) {
  return ({ 'auth/invalid-credential': 'The email or password is incorrect.', 'auth/wrong-password': 'The email or password is incorrect.', 'auth/user-not-found': 'The email or password is incorrect.', 'auth/too-many-requests': 'Too many attempts. Please wait and try again.', 'auth/email-already-in-use': 'That email already has an account. Please sign in.', 'auth/weak-password': 'Choose a stronger password.', 'auth/network-request-failed': 'Check your internet connection and try again.', 'permission-denied': 'Your account does not have access to this classroom.' })[error.code] || error.message || 'Something went wrong. Please try again.';
}

async function perform(task) {
  try { return await task(); } catch (error) { notify(friendly(error), true); }
}

function showPage(page) {
  if (!pageTitles[page]) return;
  document.querySelectorAll('[data-screen]').forEach(panel => panel.classList.toggle('hidden', panel.dataset.screen !== page));
  document.querySelectorAll('[data-page]').forEach(button => {
    button.classList.toggle('selected', button.dataset.page === page);
    if (button.dataset.page === page) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
  });
  element('pageTitle').textContent = pageTitles[page];
  window.scrollTo({ top: 0 });
}
document.querySelectorAll('[data-page]').forEach(button => button.onclick = () => showPage(button.dataset.page));
document.querySelectorAll('[data-go]').forEach(button => button.onclick = () => showPage(button.dataset.go));
element('accountButton').onclick = () => showPage('device');
document.querySelectorAll('[data-close]').forEach(button => button.onclick = () => element(button.dataset.close).close());

function timeLabel(value, date = false) {
  const time = asDate(value);
  if (!value || !Number.isFinite(time.getTime())) return '—';
  return new Intl.DateTimeFormat(undefined, date ? { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' } : { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(time);
}

function ask(title, description, accept = 'Continue') {
  return new Promise(resolve => {
    element('confirmTitle').textContent = title;
    element('confirmText').textContent = description;
    element('confirmAccept').textContent = accept;
    const dialog = element('confirmDialog');
    dialog.returnValue = 'cancel';
    element('confirmAccept').onclick = () => dialog.close('accept');
    element('confirmCancel').onclick = () => dialog.close('cancel');
    dialog.addEventListener('close', () => resolve(dialog.returnValue === 'accept'), { once: true });
    dialog.showModal();
  });
}

function watch(reference, callback) {
  const owner = generation;
  return onSnapshot(reference, snapshot => { if (owner === generation) callback(snapshot); }, error => { if (owner === generation) notify(friendly(error), true); });
}

function busy(...actions) { return actions.some(action => sending.has(action)) || [...pending.values()].some(item => actions.includes(item.action)); }
function isDeviceOnline() { return deviceOnline(deviceData, Date.now(), heartbeat, performance.now()); }
function canManage() { return isDeviceOnline() && !deviceData?.status?.running && !deviceData?.status?.initializing; }

function renderDevice() {
  const online = isDeviceOnline(), state = deviceData?.status || {};
  const connection = deviceData === null ? 'Connecting' : online ? state.sleeping ? 'Standby' : 'Online' : 'Offline';
  for (const id of ['headerConnection', 'deviceBadge', 'heroBadge']) {
    element(id).textContent = connection;
    element(id).className = id === 'heroBadge' ? 'badge light-badge' : `badge ${online ? 'success' : ''}`;
  }
  element('sidebarStatus').textContent = online ? state.sleeping ? 'Pi in standby' : 'Pi connected' : 'Pi not connected';
  element('sidebarDot').classList.toggle('online', online);
  element('deviceLabel').textContent = !online ? deviceData === null ? 'Getting things ready' : 'Your Pi is offline' : state.sleeping ? 'Taking a little break' : state.initializing ? 'Preparing your session' : state.running ? 'Attendance is live' : 'Ready for your next class';
  element('deviceDescription').textContent = state.error || (!online ? 'Keep the Python platform running on your Pi to start attendance.' : state.sleeping ? 'Wake your Pi in Device to start a new session.' : state.running ? 'Every recognized student joins the list below automatically.' : 'Choose a class, start a session, and let Face Track take the roll.');
  element('activate').disabled = !canManage() || state.sleeping || busy('start', 'stop', 'sleep', 'wake', 'settings', 'add_user', 'update_user', 'remove_user');
  element('stop').disabled = !online || !(state.running || state.initializing) || busy('start', 'stop', 'sleep');
  element('saveSettings').disabled = !canManage() || busy('settings', 'start', 'sleep');
  element('wake').disabled = !online || !state.sleeping || state.control_protocol < 2 || !state.control_protocol || busy('wake', 'sleep');
  element('sleep').disabled = !online || state.sleeping || state.control_protocol < 2 || !state.control_protocol || busy('wake', 'sleep', 'start', 'settings', 'add_user', 'update_user', 'remove_user');
  document.querySelectorAll('[data-add-student],[data-edit],[data-remove]').forEach(button => { button.disabled = !canManage() || busy('add_user', 'update_user', 'remove_user', 'start', 'settings'); });
  element('savePerson').disabled = studentSaving || !canManage() || busy('add_user', 'update_user');
  element('piCapture').disabled = !!state.sleeping;
  element('lastSeen').textContent = timeLabel(deviceData?.last_seen, true);
  element('deviceAttendance').textContent = state.running ? 'Session active' : state.initializing ? 'Starting' : 'Stopped';
  element('deviceProtocol').textContent = !online ? 'Not connected' : state.control_protocol >= 2 ? 'Teacher controls ready' : 'Update required';
  element('updateRequired').classList.toggle('hidden', !online || state.control_protocol >= 2);
  element('settingsHint').textContent = !online ? 'Connect the Pi to apply settings.' : state.running ? 'Stop attendance before changing models.' : settingsDirty ? 'You have unsaved changes.' : 'Settings apply to the next session.';
  for (const [kind, name] of [['detectors', 'detector'], ['recognizers', 'recognizer']]) {
    const available = state.capabilities?.[kind];
    for (const option of element(name).options) option.disabled = available ? !available[option.value] : false;
  }
  const active = state.session_id;
  if (followCurrent && (state.running || state.initializing) && active && sessions.some(row => row.id === active && row.state === 'running') && selected !== active) selectSession(active);
  renderSummary();
}

async function sendCommand(action, payload = {}) {
  if (!auth.currentUser || !isDeviceOnline()) throw Error('Your Pi is offline. Connect the device and try again.');
  if (['wake', 'sleep', 'update_user'].includes(action) && !(deviceData.status?.control_protocol >= 2)) throw Error('Update the Python platform on the Pi before using this control.');
  const owner = generation;
  if (busy(action)) throw Error('This request is already waiting for the Pi.');
  sending.add(action); renderDevice();
  let reference;
  try {
    reference = await addDoc(collectionFor('commands'), { action, payload, state: 'pending', created: serverTimestamp(), uid: auth.currentUser.uid });
  } finally { sending.delete(action); renderDevice(); }
  if (owner !== generation) throw Error('Signed out. Check the device before repeating the action.');
  notify('Request sent. Waiting for your Pi…');
  return new Promise((resolve, reject) => {
    const entry = { action, unsubscribe: null, timer: null, reject };
    pending.set(reference.id, entry);
    const finish = (error, result) => {
      clearTimeout(entry.timer); entry.unsubscribe?.(); pending.delete(reference.id); renderDevice();
      if (error) reject(error); else resolve(result);
    };
    entry.timer = setTimeout(() => finish(Error('No reply from the Pi yet. Check the device and history before sending this action again.')), 45000);
    entry.unsubscribe = onSnapshot(reference, snapshot => {
      if (owner !== generation) return;
      const result = snapshot.data();
      if (!result || result.state === 'pending') return;
      if (result.state !== 'done') return finish(Error(result.error || 'The Pi could not complete this request.'));
      if (action === 'start' && result.result?.session_id) { followCurrent = true; selectSession(result.result.session_id); }
      if (action === 'settings') { settingsDirty = false; populateSettings(); }
      notify(({ start: 'Session started.', stop: 'Session stopped.', add_user: 'Student saved.', update_user: 'Profile updated.', remove_user: 'Student removed. Attendance history is kept.', settings: 'Model settings saved.', wake: 'Your Pi is awake and ready.', sleep: 'Your Pi is in software standby.', approve_account: 'Teacher access approved.', reject_account: 'Request declined.' })[action] || 'Changes saved.');
      finish(null, result.result);
    }, error => finish(error));
    renderDevice();
  });
}

element('sessionForm').onsubmit = event => { event.preventDefault(); if (!element('activate').disabled) perform(() => sendCommand('start', Object.fromEntries(new FormData(event.target)))); };
element('stop').onclick = () => perform(() => sendCommand('stop'));
element('wake').onclick = () => perform(() => sendCommand('wake'));
element('sleep').onclick = () => perform(async () => { if (await ask('Put the Pi to sleep?', 'Attendance will stop and the camera will close. The Pi stays online so you can wake it here.', 'Sleep Pi')) await sendCommand('sleep'); });

function avatar(row) {
  const image = row.thumbnail;
  if (image && /^[A-Za-z0-9+/=]+$/.test(image)) return `<img class="person-avatar" src="data:image/jpeg;base64,${image}" alt="${escapeText(row.name || row.user_name || 'Student')}">`;
  const initials = String(row.name || row.user_name || '?').split(' ').filter(Boolean).map(word => word[0]).slice(0, 2).join('').toUpperCase();
  return `<span class="person-avatar" aria-hidden="true">${escapeText(initials)}</span>`;
}

function renderStudents() {
  element('studentCount').textContent = users.length;
  const matching = filterRows(users, element('studentSearch').value, ['id', 'name', 'class_name']);
  element('studentsEmpty').classList.toggle('hidden', !!users.length);
  element('studentSearchEmpty').classList.toggle('hidden', !users.length || !!matching.length);
  element('classOptions').innerHTML = [...new Set(users.map(row => row.class_name))].sort().map(name => `<option value="${escapeText(name)}"></option>`).join('');
  element('students').innerHTML = matching.map(row => `<article class="student-card">${avatar(row)}<h3>${escapeText(row.name)}</h3><p class="muted">ID ${escapeText(row.id)}${typeof row.photos === 'number' ? ` · ${row.photos} photos` : ''}</p><span class="student-class">${escapeText(row.class_name)}</span><div class="actions"><button class="secondary" data-edit="${escapeText(row.id)}">Edit profile</button><button class="danger" data-remove="${escapeText(row.id)}" aria-label="Remove ${escapeText(row.name)}">Remove</button></div></article>`).join('');
  element('students').querySelectorAll('[data-edit]').forEach(button => button.onclick = () => openStudent(users.find(row => row.id === button.dataset.edit)));
  element('students').querySelectorAll('[data-remove]').forEach(button => button.onclick = () => perform(async () => { if (await ask('Remove this student?', 'Enrolment photos will be deleted. Previous attendance records stay in history.', 'Remove student')) await sendCommand('remove_user', { id: button.dataset.remove }); }));
  renderDevice();
}
element('studentSearch').oninput = renderStudents;

function clearPhotos() { photoUrls.forEach(url => URL.revokeObjectURL(url)); photoUrls = []; element('photoPreviews').innerHTML = ''; }
function openStudent(user = null) {
  if (!canManage()) return notify('Connect the Pi and stop attendance before editing students.', true);
  editing = !!user; element('personForm').reset(); clearPhotos();
  element('personError').textContent = '';
  element('studentDialogTitle').textContent = user ? 'Edit student profile' : 'Add a student';
  element('personId').readOnly = editing;
  for (const [field, key] of [['personId', 'id'], ['personName', 'name'], ['personClass', 'class_name']]) element(field).value = user?.[key] || '';
  element('photoHint').textContent = user ? 'Existing photos are kept. Choose up to 5 more photos, or leave this empty to edit the name and class.' : 'Choose 1–5 clear photos of one student. Your phone camera is available in the file picker.';
  renderDevice(); element('studentDialog').showModal();
}
document.querySelectorAll('[data-add-student]').forEach(button => button.onclick = () => openStudent());
element('studentDialog').addEventListener('close', clearPhotos);
element('photos').onchange = () => { clearPhotos(); for (const file of [...element('photos').files].slice(0, 5)) { const url = URL.createObjectURL(file); photoUrls.push(url); const image = document.createElement('img'); image.src = url; image.alt = 'Selected enrolment photo'; element('photoPreviews').append(image); } };

async function resizedPhoto(file) {
  if (!file.type.startsWith('image/') || file.size > 20 * 1024 * 1024) throw Error('Choose image files smaller than 20 MB each.');
  const image = await createImageBitmap(file);
  try {
    const scale = Math.min(1, 720 / Math.max(image.width, image.height));
    const canvas = document.createElement('canvas'); canvas.width = Math.round(image.width * scale); canvas.height = Math.round(image.height * scale);
    canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', .78).split(',')[1];
  } finally { image.close(); }
}
element('personForm').onsubmit = async event => {
  event.preventDefault(); const owner = generation;
  if (studentSaving) return;
  studentSaving = true;
  element('personError').textContent = '';
  const button = element('savePerson'); button.disabled = true;
  try {
    const files = [...element('photos').files], capture = element('piCapture').checked;
    if (files.length > 5) throw Error('Choose at most 5 photos at once.');
    if (capture && files.length) throw Error('Choose uploaded photos or Pi capture, not both.');
    if (!editing && !capture && !files.length) throw Error('Add at least one face photo.');
    const photos = await Promise.all(files.map(resizedPhoto));
    if (photos.reduce((total, photo) => total + photo.length, 0) > 700000) throw Error('Those photos are too large together. Choose fewer photos.');
    const payload = { ...Object.fromEntries(new FormData(event.target)), photos, capture };
    await sendCommand(photos.length || capture || !editing ? 'add_user' : 'update_user', payload);
    if (owner === generation) { element('studentDialog').close(); showPage('students'); }
  } catch (error) { if (owner === generation) element('personError').textContent = friendly(error); }
  finally { studentSaving = false; if (owner === generation) renderDevice(); }
};

function populateSettings() {
  if (settingsDirty) return;
  for (const [key, value] of Object.entries(settings)) {
    const field = element('settingsForm').elements.namedItem(key);
    if (field) { if (field.type === 'checkbox') field.checked = value; else field.value = value; }
  }
}
populateSettings();
element('settingsForm').oninput = () => { settingsDirty = true; renderDevice(); };
element('settingsForm').onsubmit = event => {
  event.preventDefault();
  const values = Object.fromEntries(new FormData(event.target)); values.gate_enabled = event.target.elements.namedItem('gate_enabled').checked;
  perform(async () => { await sendCommand('settings', values); settings = { ...settings, ...values }; populateSettings(); });
};

function renderAttendance() {
  element('attendanceCount').textContent = `${attendance.length} present`;
  element('attendanceEmpty').classList.toggle('hidden', !!attendance.length);
  const session = sessions.find(row => row.id === selected);
  element('attendanceEmpty').querySelector('h3').textContent = session ? 'No attendance records yet' : 'Ready when you are';
  element('attendanceEmpty').querySelector('p').textContent = session ? 'Recognized students from this session will appear here.' : 'Start a session. Recognized students appear here automatically.';
  element('attendance').innerHTML = attendance.map((row, index) => `<article class="person-card">${avatar(row)}<div class="person-text"><strong>${escapeText(row.name || row.user_name)}</strong><small>${escapeText(row.user_id)} · ${escapeText(row.class_name)}</small></div><div class="person-time">${escapeText(timeLabel(row.timestamp))}<br><span class="badge success">Present</span></div><button class="evidence-button" data-evidence="${index}" aria-label="View recognition record for ${escapeText(row.name || row.user_name)}">↗</button></article>`).join('');
  element('attendance').querySelectorAll('[data-evidence]').forEach(button => button.onclick = () => {
    const row = attendance[Number(button.dataset.evidence)];
    const picture = row.image && /^[A-Za-z0-9+/=]+$/.test(row.image) ? `<img src="data:image/jpeg;base64,${row.image}" alt="Saved recognition image with bounding boxes">` : '';
    element('evidenceBody').innerHTML = `${picture}<p><strong>${escapeText(row.name || row.user_name)}</strong><br>ID ${escapeText(row.user_id)} · ${escapeText(row.class_name)}<br>${escapeText(timeLabel(row.timestamp, true))}<br>${escapeText(labels[row.detector])} detection / ${escapeText(labels[row.recognizer])} recognition<br>Match score: ${escapeText(row.match_score)}<br>Bounding box: ${escapeText(JSON.stringify(row.box))}</p>`;
    element('evidenceDialog').showModal();
  });
}

function renderSummary() {
  const session = sessions.find(row => row.id === selected);
  element('sessionSummary').classList.toggle('hidden', !session);
  const live = !!session && deviceData?.status?.session_id === selected && deviceData.status.running;
  if (session) {
    element('summaryName').textContent = `${session.name} · ${session.class_name || 'All classes'}`;
    element('summaryTime').textContent = timeLabel(session.started, true);
    element('summaryModel').textContent = `${labels[session.config?.detector] || '—'} / ${labels[session.config?.recognizer] || '—'}`;
    element('summaryStatus').textContent = live ? 'Active' : session.state === 'running' ? 'Awaiting device' : session.state === 'error' ? 'Error' : session.state === 'interrupted' ? 'Interrupted' : 'Completed';
    element('summaryStatus').className = `badge ${live ? 'active' : session.state === 'error' ? 'error' : ''}`;
  }
  const stats = live ? deviceData.status.statistics : session?.statistics || {};
  element('metricFps').textContent = typeof stats.mean_processing_fps === 'number' ? `${stats.mean_processing_fps.toFixed(1)} FPS` : '—';
  element('metricRecognition').textContent = typeof stats.recognition_ms === 'number' ? `${stats.recognition_ms.toFixed(1)} ms` : '—';
  element('metricDetector').textContent = labels[session?.config?.detector] || '—';
  for (const id of ['csv', 'print', 'metricsCsv', 'eventsCsv']) element(id).disabled = !selected;
}

function selectSession(id) {
  if (selected === id && attendanceSubscription) { renderSummary(); return; }
  attendanceSubscription?.(); attendanceSubscription = null; selected = id || ''; attendance = [];
  renderAttendance(); renderSummary();
  if (!selected) return;
  const ownedSession = selected;
  attendanceSubscription = watch(query(collectionFor('attendance'), where('session_id', '==', selected)), snapshot => {
    if (selected !== ownedSession) return;
    attendance = uniqueAttendance(snapshot.docs.map(item => item.data())); renderAttendance();
  });
}

function renderHistory() {
  element('sessionCount').textContent = sessions.length;
  element('historyCount').textContent = `${sessions.length} sessions`;
  const rows = filterRows(sessions, element('historySearch').value, ['name', 'class_name']);
  element('historyEmpty').classList.toggle('hidden', !!rows.length);
  element('historyList').innerHTML = rows.map(row => `<button class="history-card" data-session="${escapeText(row.id)}"><div><h3>${escapeText(row.name)}</h3><small>${escapeText(row.class_name || 'All classes')} · ${escapeText(labels[row.config?.detector] || '—')} / ${escapeText(labels[row.config?.recognizer] || '—')}</small><span class="history-date">${escapeText(timeLabel(row.started, true))}</span></div><div><span class="badge ${row.state === 'running' ? 'active' : row.state === 'error' ? 'error' : ''}">${escapeText(row.state === 'running' ? 'Session open' : row.state || 'Saved')}</span><span aria-hidden="true">↗</span></div></button>`).join('');
  element('historyList').querySelectorAll('[data-session]').forEach(button => button.onclick = () => { followCurrent = false; selectSession(button.dataset.session); showPage('home'); element('rollTitle').scrollIntoView({ block: 'start', behavior: 'auto' }); });
  element('loadHistory').classList.toggle('hidden', !hasEarlier);
  element('loadHistory').disabled = loadingHistory;
}
element('historySearch').oninput = renderHistory;
element('loadHistory').onclick = () => perform(async () => {
  if (!historyCursor || loadingHistory) return;
  loadingHistory = true; renderHistory(); const owner = generation;
  try {
    const page = await getDocs(query(collectionFor('sessions'), orderBy('started', 'desc'), startAfter(historyCursor), limit(100)));
    if (owner !== generation) return;
    historyExtra.push(...page.docs.map(item => ({ ...item.data(), id: item.id })));
    sessions = [...new Map([...sessions, ...historyExtra].map(row => [row.id, row])).values()].sort((first, second) => asDate(second.started) - asDate(first.started));
    historyCursor = page.docs.at(-1) || historyCursor; hasEarlier = page.size === 100;
  } finally { loadingHistory = false; renderHistory(); }
});

function clearSubscriptions() {
  subscriptions.forEach(unsubscribe => unsubscribe()); subscriptions = [];
  attendanceSubscription?.(); attendanceSubscription = null; accountSubscription?.(); accountSubscription = null;
  for (const entry of pending.values()) { clearTimeout(entry.timer); entry.unsubscribe?.(); entry.reject(Error('Signed out. Check the device before repeating the action.')); }
  pending.clear(); clearPhotos();
  for (const dialog of document.querySelectorAll('dialog[open]')) dialog.close();
}

async function handleAuth(user) {
  const owner = ++generation; clearSubscriptions();
  selected = ''; users = []; sessions = []; attendance = []; historyExtra = []; historyCursor = null; hasEarlier = false; deviceData = null; settingsDirty = false;
  studentSaving = false; sending.clear(); heartbeat = null;
  renderStudents(); renderAttendance(); renderHistory(); element('evidenceBody').replaceChildren();
  element('requests').replaceChildren(); element('teacherEmail').textContent = ''; element('teacherName').textContent = 'Teacher';
  settings = { ...defaults }; populateSettings(); followCurrent = true;
  element('dashboard').classList.add('hidden'); element('pendingPanel').classList.add('hidden'); element('authPanel').classList.toggle('hidden', !!user); notify('');
  if (!user) { element('password').value = ''; return; }
  try {
    const token = await user.getIdTokenResult(true);
    if (owner !== generation) return;
    if (!token.claims.face_track_admin || token.claims.face_track_device !== device) {
      element('pendingPanel').classList.remove('hidden'); element('pendingEmail').textContent = user.email;
      accountSubscription = watch(doc(database, 'face_track_accounts', user.uid), async snapshot => {
        element('requestAccess').classList.toggle('hidden', snapshot.exists());
        if (snapshot.data()?.state === 'approved') {
          const refreshed = await user.getIdTokenResult(true);
          if (owner === generation && refreshed.claims.face_track_admin && refreshed.claims.face_track_device === device) await handleAuth(user);
        } else if (snapshot.data()?.state === 'rejected') notify('Your request was declined. Contact the administrator.', true);
      });
      return;
    }
    const name = user.displayName || 'Teacher';
    element('teacherName').textContent = name; element('teacherEmail').textContent = user.email;
    element('accountButton').textContent = name[0].toUpperCase(); element('greeting').textContent = `Hello, ${name === 'Teacher' ? 'teacher' : name.split(' ')[0]} ✦`;
    element('dashboard').classList.remove('hidden'); showPage('home'); renderStudents(); renderHistory(); renderAttendance();
    subscriptions.push(watch(doc(database, 'face_track_devices', device), snapshot => {
      deviceData = snapshot.data() || {};
      heartbeat = observeHeartbeat(heartbeat, deviceData, !!(snapshot.metadata?.fromCache || snapshot.metadata?.hasPendingWrites), performance.now());
      renderDevice();
    }));
    subscriptions.push(watch(collectionFor('users'), snapshot => { users = snapshot.docs.map(item => ({ ...item.data(), id: item.id })).filter(row => row.active).sort((first, second) => first.name.localeCompare(second.name)); renderStudents(); }));
    subscriptions.push(watch(query(collectionFor('sessions'), orderBy('started', 'desc'), limit(200)), snapshot => {
      const latest = snapshot.docs.map(item => ({ ...item.data(), id: item.id }));
      sessions = [...new Map([...historyExtra, ...latest].map(row => [row.id, row])).values()].sort((first, second) => asDate(second.started) - asDate(first.started));
      if (!historyExtra.length) { historyCursor = snapshot.docs.at(-1) || null; hasEarlier = snapshot.size === 200; }
      renderHistory();
      const current = deviceData?.status?.session_id;
      const target = followCurrent && current && sessions.some(row => row.id === current) ? current : sessions.some(row => row.id === selected) ? selected : sessions[0]?.id || '';
      selectSession(target); renderDevice();
    }));
    subscriptions.push(watch(doc(database, 'face_track_devices', device, 'settings', 'current'), snapshot => { settings = { ...defaults, ...snapshot.data() }; populateSettings(); renderDevice(); }));
    subscriptions.push(watch(query(collection(database, 'face_track_accounts'), where('device_id', '==', device)), snapshot => {
      const requests = snapshot.docs.map(item => ({ ...item.data(), uid: item.id })).filter(row => row.state === 'pending');
      element('requestsEmpty').classList.toggle('hidden', !!requests.length);
      element('requests').innerHTML = requests.map(row => `<article class="person-card"><div class="person-text"><strong>${escapeText(row.name)}</strong><small>${escapeText(row.email)}</small></div><button class="secondary" data-approve="${escapeText(row.uid)}">Approve</button><button class="danger" data-reject="${escapeText(row.uid)}">Decline</button></article>`).join('');
      for (const [attribute, action] of [['approve', 'approve_account'], ['reject', 'reject_account']]) element('requests').querySelectorAll(`[data-${attribute}]`).forEach(button => button.onclick = () => perform(async () => { if (await ask('Teacher access', attribute === 'approve' ? 'Allow this teacher to manage the Pi and student data?' : 'Decline this access request?')) await sendCommand(action, { uid: button.dataset[attribute] }); }));
    }));
  } catch (error) { if (owner === generation) { element('pendingPanel').classList.remove('hidden'); notify(friendly(error), true); } }
}

element('authSwitch').onclick = () => {
  signup = !signup; element('loginForm').classList.toggle('hidden', signup); element('signupForm').classList.toggle('hidden', !signup);
  element('authTitle').textContent = signup ? 'Join your classroom.' : 'Hello, teacher.';
  element('authDescription').textContent = signup ? 'Request permission to manage this attendance device.' : 'Sign in to your classroom workspace.';
  element('authSwitch').textContent = signup ? 'Already have an account? Sign in' : 'New here? Request a teacher account'; notify('');
};
element('loginForm').onsubmit = event => { event.preventDefault(); const button = event.target.querySelector('button'); button.disabled = true; perform(() => signInWithEmailAndPassword(auth, element('email').value.trim(), element('password').value)).finally(() => { button.disabled = false; }); };
async function requestAccount(user) {
  const reference = doc(database, 'face_track_accounts', user.uid), snapshot = await getDoc(reference);
  if (!snapshot.exists()) await setDoc(reference, { uid: user.uid, email: user.email, name: registration || user.displayName || user.email, device_id: device, state: 'pending', created: serverTimestamp() });
}
element('signupForm').onsubmit = async event => {
  event.preventDefault(); if (element('signupPassword').value !== element('signupConfirm').value) return notify('The passwords do not match.', true);
  const button = event.target.querySelector('button'); button.disabled = true; registration = element('signupName').value.trim();
  try { const result = await createUserWithEmailAndPassword(auth, element('signupEmail').value.trim(), element('signupPassword').value); await updateProfile(result.user, { displayName: registration }); await requestAccount(result.user); event.target.reset(); }
  catch (error) { notify(friendly(error), true); } finally { registration = null; button.disabled = false; }
};
element('requestAccess').onclick = () => perform(async () => { await requestAccount(auth.currentUser); notify('Access request sent.'); });
element('refreshAccess').onclick = () => perform(() => handleAuth(auth.currentUser));
element('logout').onclick = element('pendingLogout').onclick = () => perform(() => signOut(auth));

function download(rows, filename) {
  const url = URL.createObjectURL(new Blob([csvText(rows)], { type: 'text/csv;charset=utf-8' }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
element('csv').onclick = () => perform(() => download(attendance, `attendance-${selected}.csv`));
element('metricsCsv').onclick = () => perform(async () => {
  const records = [...attendance]; const rows = [];
  for (const row of records) { const snapshot = await getDoc(doc(database, 'face_track_devices', device, 'frames', String(row.frame_id))); if (snapshot.exists()) rows.push({ user_id: row.user_id, name: row.name, ...snapshot.data() }); }
  download(rows, 'attendance-statistics.csv');
});
element('eventsCsv').onclick = () => perform(async () => {
  const sessionId = selected, rows = []; let cursor = null;
  while (true) {
    const conditions = [where('session_id', '==', sessionId), orderBy('timestamp'), limit(500)];
    if (cursor) conditions.push(startAfter(cursor));
    const page = await getDocs(query(collectionFor('events'), ...conditions)); rows.push(...page.docs.map(item => item.data()));
    if (page.size < 500) break; cursor = page.docs.at(-1);
  }
  download(rows, 'recognition-events.csv');
});
element('print').onclick = () => {
  const popup = window.open('', '_blank'); if (!popup) return notify('Allow popups to open a printable report.', true);
  const session = sessions.find(row => row.id === selected);
  const rows = attendance.map(row => `<tr><td>${escapeText(timeLabel(row.timestamp, true))}</td><td>${escapeText(row.user_id)}</td><td>${escapeText(row.name)}</td><td>${escapeText(row.class_name)}</td><td>${escapeText(labels[row.detector])} / ${escapeText(labels[row.recognizer])}</td><td>${row.image && /^[A-Za-z0-9+/=]+$/.test(row.image) ? `<img width="140" src="data:image/jpeg;base64,${row.image}" alt="Recognition evidence">` : ''}</td></tr>`).join('');
  popup.document.write(`<!doctype html><html><head><title>Face Track attendance</title><style>body{font:14px Arial;padding:25px;color:#173a35}table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}button{padding:10px}@media print{button{display:none}}</style></head><body><h1>${escapeText(session?.name || 'Attendance')}</h1><p>${escapeText(session?.class_name || 'All classes')} · ${escapeText(timeLabel(session?.started, true))} · ${attendance.length} present</p><button onclick="window.print()">Print / Save PDF</button><table><thead><tr><th>Time</th><th>ID</th><th>Student</th><th>Class</th><th>Models</th><th>Evidence</th></tr></thead><tbody>${rows}</tbody></table></body></html>`); popup.document.close(); popup.opener = null;
};
element('dateLabel').textContent = new Intl.DateTimeFormat(undefined, { weekday: 'long', month: 'long', day: 'numeric' }).format(new Date()).toUpperCase();
element('deviceName').textContent = element('sidebarDevice').textContent = device;
onAuthStateChanged(auth, handleAuth);
setInterval(() => { if (!element('dashboard').classList.contains('hidden')) renderDevice(); }, 5000);
