import { initializeApp } from 'https://www.gstatic.com/firebasejs/10.12.5/firebase-app.js';
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, updateProfile, onAuthStateChanged, signOut } from 'https://www.gstatic.com/firebasejs/10.12.5/firebase-auth.js';
import { getFirestore, collection, doc, onSnapshot, query, orderBy, limit, where, addDoc, setDoc, getDoc, getDocs, serverTimestamp, startAfter } from 'https://www.gstatic.com/firebasejs/10.12.5/firebase-firestore.js';

const app = initializeApp(window.FIREBASE_CONFIG);
const auth = getAuth(app);
const database = getFirestore(app);
const device = window.DEVICE_ID;
const element = id => document.getElementById(id);
const collectionFor = name => collection(database, 'face_track_devices', device, name);
const escapeText = value => String(value ?? '').replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
const names = { haar: 'Haar Cascade', yunet: 'YuNet', dlib: 'dlib', lbph: 'LBPH', sface: 'SFace' };
let subscriptions = [];
let attendanceSubscription = null;
let pendingSubscription = null;
let selected = '';
let sessions = [];
let attendance = [];
let deviceData = null;
let settings = {};
let followCurrent = true;
let registration = null;
let settingsDirty = false;
let authGeneration = 0;
const queued = new Map();

function notify(text, failure = false) {
  element('message').textContent = text;
  element('message').classList.toggle('error', failure);
  element('message').classList.toggle('hidden', !text);
}

function friendlyError(error) {
  const messages = {
    'auth/email-already-in-use': 'This email already has an account. Please sign in.',
    'auth/invalid-email': 'Enter a valid email address.',
    'auth/weak-password': 'Choose a stronger password with at least 8 characters.',
    'auth/invalid-credential': 'The email or password is incorrect.',
    'auth/wrong-password': 'The email or password is incorrect.',
    'auth/user-not-found': 'The email or password is incorrect.',
    'auth/too-many-requests': 'Too many attempts. Wait a moment and try again.',
    'permission-denied': 'Your account does not have access to this device.',
    'auth/network-request-failed': 'Could not connect. Check your internet connection.'
  };
  return messages[error.code] || error.message || 'Something went wrong. Please try again.';
}

function timestamp(value) {
  return value?.toDate ? value.toDate() : new Date(value || 0);
}

function timeLabel(value, includeDate = false) {
  if (!value) return '—';
  const date = timestamp(value);
  if (Number.isNaN(date.getTime())) return '—';
  return new Intl.DateTimeFormat(undefined, includeDate
    ? { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }
    : { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(date);
}

function watch(reference, callback) {
  return onSnapshot(reference, callback, error => notify(friendlyError(error), true));
}

function online() {
  return !!deviceData?.last_seen && Math.abs(Date.now() - timestamp(deviceData.last_seen).getTime()) < 20000;
}

function renderDevice() {
  const known = deviceData !== null;
  const connected = online();
  const state = deviceData?.status || {};
  const busy = [...queued.values()].some(action => action === 'start' || action === 'stop');
  element('connectionBadge').textContent = !known ? 'Connecting' : connected ? 'Online' : 'Offline';
  element('connectionBadge').className = 'badge ' + (connected ? 'online' : 'neutral');
  element('deviceLabel').textContent = !known ? 'Connecting…' : !connected ? 'Device offline' : state.initializing ? 'Preparing attendance' : state.running ? 'Attendance active' : 'Ready to start';
  element('deviceDescription').textContent = state.error || (!known ? 'Checking the device connection.' : !connected
    ? 'Connect the attendance device to manage a session.'
    : state.running ? 'Recognized people appear below automatically.' : 'Activate a new session or take one attendance shot.');
  element('activate').disabled = !connected || busy || !!state.running || !!state.initializing;
  element('stop').disabled = !connected || busy || !(state.running || state.initializing);
  element('saveSettings').disabled = !connected || !!state.running || !!state.initializing || [...queued.values()].includes('settings');
  for (const id of ['addPerson', 'capture']) {
    element(id).disabled = !connected || !!state.running || !!state.initializing || [...queued.values()].includes('add_user');
  }
  for (const [kind, id] of [['detectors', 'detector'], ['recognizers', 'recognizer']]) {
    const available = state.capabilities?.[kind];
    for (const option of element(id).options) option.disabled = available ? !available[option.value] : false;
  }
  const current = state.session_id;
  if (followCurrent && current && sessions.some(session => session.id === current) && selected !== current) selectSession(current);
  renderSummary();
}

async function sendCommand(action, payload) {
  if (!online()) throw Error('The attendance device is offline. Connect it and try again.');
  const reference = await addDoc(collectionFor('commands'), {
    action, payload, state: 'pending', created: serverTimestamp(), uid: auth.currentUser.uid
  });
  queued.set(reference.id, action);
  renderDevice();
  notify(action === 'start' ? 'Activating attendance…' : action === 'stop' ? 'Deactivating attendance…' : 'Sending your changes…');
  let unsubscribe = null;
  unsubscribe = watch(reference, snapshot => {
    const result = snapshot.data();
    if (!result || result.state === 'pending') return;
    queued.delete(reference.id);
    if (result.state === 'done') {
      const messages = { start: 'Attendance activated.', stop: 'Attendance deactivated.', settings: 'Settings saved.', add_user: 'Person saved.', remove_user: 'Person removed.', approve_account: 'Account approved.', reject_account: 'Request declined.' };
      notify(messages[action] || 'Changes saved.');
      if (action === 'start' && result.result?.session_id) {
        followCurrent = true;
        selectSession(result.result.session_id);
      }
      if (action === 'settings') settingsDirty = false;
    } else notify(result.error || 'The device could not complete this request.', true);
    renderDevice();
    if (unsubscribe) unsubscribe();
  });
  subscriptions.push(() => unsubscribe?.());
}

async function perform(task) {
  try { await task(); } catch (error) { notify(friendlyError(error), true); }
}

document.querySelectorAll('[data-tab]').forEach(button => button.onclick = () => {
  for (const panel of ['attendanceTab', 'peopleTab', 'settingsTab']) element(panel).classList.toggle('hidden', panel !== button.dataset.tab);
  document.querySelectorAll('[data-tab]').forEach(tab => {
    const active = tab === button;
    tab.classList.toggle('selected', active);
    tab.setAttribute('aria-selected', String(active));
  });
});

function authTab(signup) {
  element('loginForm').classList.toggle('hidden', signup);
  element('signupForm').classList.toggle('hidden', !signup);
  element('showLogin').classList.toggle('selected', !signup);
  element('showSignup').classList.toggle('selected', signup);
  element('authTitle').textContent = signup ? 'Create your account' : 'Welcome back';
  element('authDescription').textContent = signup ? 'Request access to manage attendance.' : 'Sign in to manage your attendance device.';
  notify('');
}
element('showLogin').onclick = () => authTab(false);
element('showSignup').onclick = () => authTab(true);
element('logout').onclick = () => perform(() => signOut(auth));
element('loginForm').onsubmit = event => {
  event.preventDefault();
  perform(() => signInWithEmailAndPassword(auth, element('email').value.trim(), element('password').value));
};

async function requestAccount(user) {
  const reference = doc(database, 'face_track_accounts', user.uid);
  const existing = await getDoc(reference);
  if (!existing.exists()) {
    await setDoc(reference, { uid: user.uid, email: user.email, name: registration?.name || user.displayName || user.email,
      device_id: device, state: 'pending', created: serverTimestamp() });
  }
}

element('signupForm').onsubmit = async event => {
  event.preventDefault();
  if (element('signupPassword').value !== element('signupConfirm').value) return notify('The passwords do not match.', true);
  const button = event.target.querySelector('button');
  button.disabled = true;
  registration = { name: element('signupName').value.trim() };
  try {
    const result = await createUserWithEmailAndPassword(auth, element('signupEmail').value.trim(), element('signupPassword').value);
    await updateProfile(result.user, { displayName: registration.name });
    await requestAccount(result.user);
    element('signupForm').reset();
  } catch (error) { notify(friendlyError(error), true); }
  finally { registration = null; button.disabled = false; }
};

element('requestAccess').onclick = () => perform(async () => {
  await requestAccount(auth.currentUser);
  element('requestAccess').classList.add('hidden');
  notify('Your access request has been sent.');
});
element('refreshAccess').onclick = () => perform(async () => {
  const token = await auth.currentUser.getIdTokenResult(true);
  if (token.claims.face_track_admin && token.claims.face_track_device === device) await handleAuth(auth.currentUser);
  else notify('Your account is still waiting for approval.');
});

element('sessionForm').onsubmit = event => {
  event.preventDefault();
  followCurrent = true;
  perform(() => sendCommand('start', Object.fromEntries(new FormData(event.target))));
};
element('stop').onclick = () => perform(() => sendCommand('stop', {}));
element('settingsForm').addEventListener('input', () => { settingsDirty = true; });
element('settingsForm').onsubmit = event => {
  event.preventDefault();
  const values = Object.fromEntries(new FormData(event.target));
  values.gate_enabled = event.target.gate_enabled.checked;
  perform(() => sendCommand('settings', values));
};

function resizedPhoto(file) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      const scale = Math.min(1, 640 / image.width, 640 / image.height);
      const canvas = document.createElement('canvas');
      canvas.width = Math.round(image.width * scale);
      canvas.height = Math.round(image.height * scale);
      canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL('image/jpeg', .8).split(',')[1]);
    };
    image.onerror = () => { URL.revokeObjectURL(url); reject(Error('One of the photos cannot be opened.')); };
    image.src = url;
  });
}

element('personForm').onsubmit = event => {
  event.preventDefault();
  perform(async () => {
    const files = [...element('photos').files];
    if (!files.length || files.length > 5) throw Error('Select between 1 and 5 photos.');
    const photos = await Promise.all(files.map(resizedPhoto));
    if (photos.reduce((total, photo) => total + photo.length, 0) > 700000) throw Error('Please upload fewer photos at once.');
    await sendCommand('add_user', { ...Object.fromEntries(new FormData(event.target)), photos });
  });
};
element('capture').onclick = () => {
  if (element('personForm').reportValidity()) perform(() => sendCommand('add_user', { ...Object.fromEntries(new FormData(element('personForm'))), capture: true }));
};

function avatar(row, field = 'thumbnail') {
  const image = row[field];
  if (image && /^[A-Za-z0-9+/=]+$/.test(image)) return `<img class="person-avatar" src="data:image/jpeg;base64,${image}" alt="${escapeText(row.name || row.user_name || 'Face')}">`;
  const initials = (row.name || row.user_name || '?').split(' ').map(word => word[0]).slice(0, 2).join('').toUpperCase();
  return `<span class="person-avatar" aria-hidden="true">${escapeText(initials)}</span>`;
}

function renderAttendance() {
  element('attendanceCount').textContent = `${attendance.length} present`;
  element('attendanceEmpty').classList.toggle('hidden', attendance.length > 0);
  element('attendance').innerHTML = attendance.map(row => `<article class="person-card" data-user-id="${escapeText(row.user_id)}">
    ${avatar(row)}<div class="person-text"><strong>${escapeText(row.name || row.user_name)}</strong><small>${escapeText(row.user_id)} · ${escapeText(row.class_name)}</small></div>
    <div class="person-time">${escapeText(timeLabel(row.timestamp))}<br><span class="badge online">Present</span></div></article>`).join('');
}

function renderSummary() {
  const session = sessions.find(row => row.id === selected);
  element('sessionSummary').classList.toggle('hidden', !session);
  if (session) {
    element('summaryClass').textContent = session.class_name || 'All classes';
    element('summaryTime').textContent = timeLabel(session.started, true);
    const live = deviceData?.status?.session_id === selected && deviceData?.status?.running;
    element('summaryStatus').textContent = live ? 'Active' : session.state === 'error' ? 'Failed' : session.state === 'running' ? 'Preparing' : 'Completed';
    element('summaryStatus').className = 'badge ' + (live ? 'active' : 'neutral');
  }
  const stats = deviceData?.status?.session_id === selected && deviceData.status.running
    ? deviceData.status.statistics : session?.statistics || {};
  element('metricFps').textContent = typeof stats?.mean_processing_fps === 'number' ? `${stats.mean_processing_fps.toFixed(1)} FPS` : '—';
  element('metricRecognition').textContent = typeof stats?.recognition_ms === 'number' ? `${stats.recognition_ms.toFixed(1)} ms` : '—';
  element('metricDetector').textContent = names[session?.config?.detector] || '—';
  for (const id of ['csv', 'print', 'metricsCsv', 'eventsCsv']) element(id).disabled = !selected;
}

function selectSession(sessionId) {
  if (selected === sessionId && attendanceSubscription) { renderSummary(); return; }
  attendanceSubscription?.();
  attendanceSubscription = null;
  selected = sessionId || '';
  attendance = [];
  element('sessions').value = selected;
  renderAttendance();
  renderSummary();
  if (!selected) return;
  attendanceSubscription = watch(query(collectionFor('attendance'), where('session_id', '==', selected)), snapshot => {
    const people = new Map();
    snapshot.docs.map(document => document.data()).sort((first, second) => first.timestamp.localeCompare(second.timestamp))
      .forEach(row => { if (!people.has(row.user_id)) people.set(row.user_id, row); });
    attendance = [...people.values()];
    renderAttendance();
  });
}

element('sessions').onchange = () => { followCurrent = false; selectSession(element('sessions').value); };

function clearSubscriptions() {
  subscriptions.forEach(unsubscribe => unsubscribe());
  subscriptions = [];
  attendanceSubscription?.(); attendanceSubscription = null;
  pendingSubscription?.(); pendingSubscription = null;
  queued.clear();
}

async function handleAuth(user) {
  const generation = ++authGeneration;
  clearSubscriptions();
  element('dashboard').classList.add('hidden');
  element('pendingPanel').classList.add('hidden');
  element('authPanel').classList.toggle('hidden', !!user);
  element('logout').classList.toggle('hidden', !user);
  notify('');
  if (!user) { selected = ''; attendance = []; sessions = []; deviceData = null; settingsDirty = false; return; }
  try {
    const token = await user.getIdTokenResult(true);
    if (generation !== authGeneration) return;
    if (!token.claims.face_track_admin || token.claims.face_track_device !== device) {
      element('pendingPanel').classList.remove('hidden');
      element('pendingEmail').textContent = user.email;
      pendingSubscription = watch(doc(database, 'face_track_accounts', user.uid), async snapshot => {
        element('requestAccess').classList.toggle('hidden', snapshot.exists());
        if (snapshot.data()?.state === 'approved') {
          const updated = await user.getIdTokenResult(true);
          if (generation === authGeneration && updated.claims.face_track_admin && updated.claims.face_track_device === device) await handleAuth(user);
        } else if (snapshot.data()?.state === 'rejected') notify('Your access request was declined. Contact the device admin.', true);
      });
      return;
    }
    element('dashboard').classList.remove('hidden');
    subscriptions.push(watch(doc(database, 'face_track_devices', device), snapshot => {
      deviceData = snapshot.data() || {};
      renderDevice();
    }));
    subscriptions.push(watch(collectionFor('users'), snapshot => {
      const users = snapshot.docs.map(document => document.data()).filter(row => row.active);
      element('peopleCount').textContent = `${users.length} people`;
      element('peopleEmpty').classList.toggle('hidden', !!users.length);
      element('users').innerHTML = users.map(row => `<article class="person-card">${avatar(row)}<div class="person-text"><strong>${escapeText(row.name)}</strong><small>${escapeText(row.id)} · ${escapeText(row.class_name)}</small></div><button class="danger" data-remove="${escapeText(row.id)}">Remove</button></article>`).join('');
      document.querySelectorAll('[data-remove]').forEach(button => button.onclick = () => {
        if (confirm('Remove this person? Previous attendance will be kept.')) perform(() => sendCommand('remove_user', { id: button.dataset.remove }));
      });
    }));
    subscriptions.push(watch(query(collectionFor('sessions'), orderBy('started', 'desc'), limit(200)), snapshot => {
      sessions = snapshot.docs.map(document => ({ ...document.data(), id: document.id }));
      element('sessions').innerHTML = sessions.length ? sessions.map(row => `<option value="${escapeText(row.id)}">${escapeText(row.name)} · ${escapeText(timeLabel(row.started, true))}</option>`).join('') : '<option value="">No sessions yet</option>';
      const active = deviceData?.status?.session_id;
      const target = followCurrent && active && sessions.some(row => row.id === active) ? active
        : sessions.some(row => row.id === selected) ? selected : sessions[0]?.id || '';
      selectSession(target);
      element('sessions').value = selected;
    }));
    subscriptions.push(watch(doc(database, 'face_track_devices', device, 'settings', 'current'), snapshot => {
      settings = snapshot.data() || {};
      if (!settingsDirty) {
        const form = element('settingsForm');
        for (const [key, value] of Object.entries(settings)) {
          const field = form.elements.namedItem(key);
          if (field) { if (field.type === 'checkbox') field.checked = value; else field.value = value; }
        }
      }
    }));
    subscriptions.push(watch(query(collection(database, 'face_track_accounts'), where('device_id', '==', device)), snapshot => {
      const requests = snapshot.docs.map(document => ({ ...document.data(), uid: document.id })).filter(row => row.state === 'pending');
      element('requestsEmpty').classList.toggle('hidden', !!requests.length);
      element('requests').innerHTML = requests.map(row => `<article class="person-card"><div class="person-text"><strong>${escapeText(row.name)}</strong><small>${escapeText(row.email)}</small></div><button class="primary" data-approve="${escapeText(row.uid)}">Approve</button><button class="danger" data-reject="${escapeText(row.uid)}">Decline</button></article>`).join('');
      for (const [attribute, action] of [['approve', 'approve_account'], ['reject', 'reject_account']]) document.querySelectorAll(`[data-${attribute}]`).forEach(button => button.onclick = () => {
        if (confirm(attribute === 'approve' ? 'Give this account permission to manage this device?' : 'Decline this access request?')) perform(() => sendCommand(action, { uid: button.dataset[attribute] }));
      });
    }));
    renderDevice();
  } catch (error) { notify(friendlyError(error), true); }
}

onAuthStateChanged(auth, user => handleAuth(user));
setInterval(() => { if (!element('dashboard').classList.contains('hidden')) renderDevice(); }, 5000);

function downloadCsv(rows, filename) {
  if (!rows.length) throw Error('There are no records to export.');
  const keys = Object.keys(rows[0]).filter(key => !['image', 'thumbnail'].includes(key));
  const escapeCell = value => {
    let text = typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value ?? '');
    if (/^[=+@-]/.test(text)) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  };
  const text = [keys.map(escapeCell), ...rows.map(row => keys.map(key => escapeCell(row[key])))].map(row => row.join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob(['\ufeff' + text], { type: 'text/csv;charset=utf-8' }));
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = filename; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
element('csv').onclick = () => perform(() => downloadCsv(attendance, 'attendance.csv'));
element('metricsCsv').onclick = () => perform(async () => {
  const rows = [];
  for (const person of attendance) {
    const snapshot = await getDoc(doc(database, 'face_track_devices', device, 'frames', String(person.frame_id)));
    if (snapshot.exists()) rows.push({ user_id: person.user_id, name: person.name, ...snapshot.data() });
  }
  downloadCsv(rows, 'attendance-statistics.csv');
});
element('eventsCsv').onclick = () => perform(async () => {
  const sessionId = selected;
  let cursor = null;
  const rows = [];
  while (true) {
    const conditions = [where('session_id', '==', sessionId), orderBy('timestamp'), limit(500)];
    if (cursor) conditions.push(startAfter(cursor));
    const page = await getDocs(query(collectionFor('events'), ...conditions));
    rows.push(...page.docs.map(document => document.data()));
    if (page.size < 500) break;
    cursor = page.docs[page.docs.length - 1];
  }
  downloadCsv(rows, 'recognition-events.csv');
});
element('print').onclick = () => {
  const popup = window.open('', '_blank');
  if (!popup) return notify('Allow popups to print the report.', true);
  const rows = attendance.map(row => `<tr><td>${escapeText(timeLabel(row.timestamp, true))}</td><td>${escapeText(row.user_id)}</td><td>${escapeText(row.name)}</td><td>${escapeText(row.class_name)}</td><td>${escapeText(names[row.detector])} / ${escapeText(names[row.recognizer])}</td><td>${row.image && /^[A-Za-z0-9+/=]+$/.test(row.image) ? `<img width="180" src="data:image/jpeg;base64,${row.image}">` : ''}</td></tr>`).join('');
  popup.document.write(`<title>Attendance report</title><style>body{font-family:Arial}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:10px}tr{break-inside:avoid}</style><h1>Attendance report</h1><table><tr><th>Time</th><th>ID</th><th>Name</th><th>Class</th><th>Models</th><th>Image</th></tr>${rows}</table>`);
  popup.document.close();
  setTimeout(() => popup.print(), 1000);
};
