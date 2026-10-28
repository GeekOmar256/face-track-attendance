import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { JSDOM } from 'jsdom';
import * as utilities from '../public/app-utils.mjs';

function dashboard() {
  const html = readFileSync(new URL('../public/index.html', import.meta.url), 'utf8');
  const code = readFileSync(new URL('../public/dashboard.js', import.meta.url), 'utf8').replace(/^import .+;\r?$/gm, '');
  const dom = new JSDOM(html, { url: 'https://classroom.example/', runScripts: 'outside-only' });
  const window = dom.window, listeners = new Map(), commands = [];
  const auth = { currentUser: null };
  let authChanged;
  const ref = (...parts) => ({ path: parts.filter(item => typeof item === 'string').join('/'), id: parts.at(-1) });
  Object.assign(window, utilities, {
    FIREBASE_CONFIG: {}, DEVICE_ID: 'classroom-pi', scrollTo() {},
    initializeApp: () => ({}), getAuth: () => auth, getFirestore: () => ({}),
    collection: ref, doc: ref, query: (reference, ...constraints) => ({ ...reference, constraints }),
    where: (...args) => args, orderBy: (...args) => args, limit: (...args) => args, startAfter: (...args) => args,
    serverTimestamp: () => 'server-time',
    onAuthStateChanged: (_, callback) => { authChanged = callback; },
    onSnapshot: (reference, callback) => {
      const set = listeners.get(reference.path) || new Set(); set.add(callback); listeners.set(reference.path, set);
      return () => set.delete(callback);
    },
    addDoc: async (reference, data) => { const document = ref(reference.path, `command-${commands.length}`); commands.push({ document, data }); return document; },
    signOut: async () => { auth.currentUser = null; await authChanged(null); },
    signInWithEmailAndPassword: async () => {}, createUserWithEmailAndPassword: async () => {}, updateProfile: async () => {},
    setDoc: async () => {}, getDoc: async () => ({ exists: () => false }), getDocs: async () => ({ docs: [], size: 0 })
  });
  window.HTMLElement.prototype.scrollIntoView = () => {};
  window.HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
  window.HTMLDialogElement.prototype.close = function (value = '') { this.returnValue = value; this.removeAttribute('open'); this.dispatchEvent(new window.Event('close')); };
  vm.runInContext(code, dom.getInternalVMContext());
  const emit = (path, data) => {
    const snapshot = Array.isArray(data)
      ? { docs: data.map(row => ({ id: row.id, data: () => row })), size: data.length }
      : { exists: () => !!data, data: () => data };
    for (const callback of [...listeners.get(path) || []]) callback(snapshot);
  };
  const root = 'face_track_devices/classroom-pi';
  return {
    window, commands, auth, emit, root, close: () => dom.window.close(),
    el: id => window.document.getElementById(id),
    online: (extra = {}) => emit(root, { last_seen: new Date().toISOString(), status: { running: false, control_protocol: 2, ...extra } }),
    signIn: async () => {
      auth.currentUser = { uid: 'teacher', email: 'teacher@example.invalid', displayName: 'Teacher', getIdTokenResult: async () => ({ claims: { face_track_admin: true, face_track_device: 'classroom-pi' } }) };
      await authChanged(auth.currentUser);
    },
    done: () => { const command = commands.at(-1); emit(command.document.path, { state: 'done', result: {} }); }
  };
}

const tick = () => new Promise(resolve => setImmediate(resolve));

test('teacher navigation, connectivity, attendance updates and sign-out clearing', async t => {
  const app = dashboard(); t.after(app.close);
  await app.signIn();
  assert.equal(app.el('dashboard').classList.contains('hidden'), false);
  assert.equal(app.el('activate').disabled, true);
  app.online();
  assert.equal(app.el('activate').disabled, false);
  assert.equal(app.el('sleep').disabled, false);
  for (const page of ['students', 'history', 'models', 'device', 'home']) {
    app.window.document.querySelector(`[data-page="${page}"]`).click();
    assert.equal(app.window.document.querySelector(`[data-screen="${page}"]`).classList.contains('hidden'), false);
  }
  app.emit(`${app.root}/users`, [{ id: '101', name: 'Mona <student>', class_name: 'CE492', active: true, photos: 2 }]);
  assert.equal(app.el('studentCount').textContent, '1');
  assert.equal(app.el('students').querySelector('student'), null);
  app.emit(`${app.root}/sessions`, [{ id: 's1', name: 'Morning', class_name: 'CE492', state: 'completed', started: '2026-10-28T08:00:00Z', config: { detector: 'haar', recognizer: 'lbph' } }]);
  const row = { id: 'a1', session_id: 's1', user_id: '101', name: 'Mona', class_name: 'CE492', timestamp: '2026-10-28T08:01:00Z' };
  app.emit(`${app.root}/attendance`, [row, { ...row, id: 'a2', timestamp: '2026-10-28T08:02:00Z' }]);
  assert.equal(app.el('attendanceCount').textContent, '1 present');
  app.emit(`${app.root}/attendance`, [row, { ...row, id: 'a3', user_id: '102', name: 'Nour' }]);
  assert.equal(app.el('attendance').children.length, 2);
  app.emit(app.root, { last_seen: '2020-01-01', status: { running: true } });
  assert.equal(app.el('headerConnection').textContent, 'Offline');
  assert.equal(app.el('activate').disabled, true);
  await app.el('logout').onclick(); await tick();
  assert.equal(app.el('students').children.length, 0);
  assert.equal(app.el('attendance').children.length, 0);
  assert.equal(app.el('dashboard').classList.contains('hidden'), true);
});

test('student edit waits for Pi acknowledgement and preserves the student ID', async t => {
  const app = dashboard(); t.after(app.close);
  await app.signIn(); app.online();
  app.emit(`${app.root}/users`, [{ id: '101', name: 'Mona', class_name: 'CE492', active: true }]);
  app.el('students').querySelector('[data-edit]').click();
  assert.equal(app.el('personId').readOnly, true);
  app.el('personName').value = 'Mona Updated';
  const save = app.el('personForm').onsubmit({ preventDefault() {}, target: app.el('personForm') });
  await tick();
  assert.equal(app.commands.length, 1);
  assert.equal(app.commands[0].data.action, 'update_user');
  assert.equal(app.commands[0].data.payload.id, '101');
  assert.equal(app.commands[0].data.payload.name, 'Mona Updated');
  assert.equal(app.el('studentDialog').hasAttribute('open'), true);
  assert.equal(app.el('savePerson').disabled, true);
  app.done(); await save;
  assert.equal(app.el('studentDialog').hasAttribute('open'), false);
  assert.equal(app.el('message').textContent, 'Profile updated.');
});

test('new students require photos and sleep requires confirmation', async t => {
  const app = dashboard(); t.after(app.close);
  await app.signIn(); app.online();
  app.window.document.querySelector('[data-add-student]').click();
  app.el('personId').value = '102'; app.el('personName').value = 'Nour'; app.el('personClass').value = 'CE492';
  await app.el('personForm').onsubmit({ preventDefault() {}, target: app.el('personForm') });
  assert.match(app.el('personError').textContent, /at least one face photo/);
  assert.equal(app.commands.length, 0);
  app.el('studentDialog').close();
  const sleep = app.el('sleep').onclick();
  assert.equal(app.commands.length, 0);
  app.el('confirmAccept').click(); await tick();
  assert.equal(app.commands[0].data.action, 'sleep');
  app.done(); await sleep;
  assert.match(app.el('message').textContent, /software standby/);
});

test('model selection and one-shot sessions send real device commands', async t => {
  const app = dashboard(); t.after(app.close);
  await app.signIn(); app.online({ capabilities: { detectors: { haar: true, yunet: true, dlib: false }, recognizers: { lbph: true, sface: true, dlib: false } } });
  assert.equal(app.el('detector').querySelector('option[value="dlib"]').disabled, true);
  app.el('detector').value = 'yunet'; app.el('recognizer').value = 'sface'; app.el('gate').checked = false;
  app.el('settingsForm').onsubmit({ preventDefault() {}, target: app.el('settingsForm') }); await tick();
  assert.equal(app.commands[0].data.action, 'settings');
  assert.equal(app.commands[0].data.payload.detector, 'yunet');
  assert.equal(app.commands[0].data.payload.recognizer, 'sface');
  assert.equal(app.commands[0].data.payload.gate_enabled, false);
  app.done(); await tick();
  app.el('sessionName').value = 'Morning class'; app.el('sessionClass').value = 'CE492';
  app.window.document.querySelector('[name="mode"][value="single"]').checked = true;
  app.el('sessionForm').onsubmit({ preventDefault() {}, target: app.el('sessionForm') }); await tick();
  assert.equal(app.commands[1].data.action, 'start');
  assert.equal(app.commands[1].data.payload.mode, 'single');
  assert.equal(app.commands[1].data.payload.class_name, 'CE492');
  app.done(); await tick();
});
