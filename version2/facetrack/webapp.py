"""Browser front end for the Face Track pipeline.

Built on Python's own http.server, so it adds no dependency at all: the
Raspberry Pi still needs nothing beyond opencv-contrib-python and numpy. The
page is served to any browser on the same Wi-Fi, which matches the application
layer described in the FYP1 plan, where the dashboard reaches the Raspberry Pi
over the local network.

It also means no display is needed on the Pi itself. A headless board with no
desktop installed can serve this page, and the video arrives as JPEG frames
over HTTP rather than as X11 traffic forwarded through SSH.

Version 2 adds the second tab: enrolling a new person with their details, and
retraining a recognizer, without leaving the browser.
"""

from __future__ import annotations

import csv
import io
import json
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from . import config
from .detectors import DETECTOR_NAMES, get_detector
from .enrollment import DETAIL_FIELDS, EnrollmentSession, delete_person, list_people
from .recognizers import RECOGNIZER_NAMES, default_model_path
from .session import PipelineSession
from .training import TrainingJob

BOUNDARY = "facetrackframe"
UPLOAD_DIR = config.DATA_DIR / "uploads"
MAX_UPLOAD = 512 * 1024 * 1024          # 512 MB, enough for a classroom video


# ======================================================================= page
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Track</title>
<style>
  :root {
    --bg:#f4f7fb; --panel:#ffffff; --soft:#f7f9fc; --line:#e2e8f2;
    --text:#26344a; --muted:#7b8aa3;
    --accent:#4c7ef3; --accent-soft:#e9f0ff; --accent-deep:#2f5fd0;
    --ok:#1f9d55; --ok-soft:#e4f6ea;
    --bad:#dc5563; --bad-soft:#fdecee;
    --warn:#dd8c1f; --warn-soft:#fdf2e2;
    --shadow:0 1px 2px rgba(38,52,74,.06), 0 6px 18px rgba(38,52,74,.06);
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  img { max-width:100%; }

  header { background:var(--panel); border-bottom:1px solid var(--line);
           padding:12px 18px 0; }
  .brand { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
  .brand h1 { margin:0; font-size:18px; letter-spacing:-.2px; }
  .brand span { color:var(--muted); font-size:12.5px; }
  .brand .state { margin-left:auto; }

  .tabs { display:flex; gap:4px; margin-top:12px; }
  .tabs button { border:0; background:none; padding:9px 16px; cursor:pointer;
                 font:inherit; font-weight:600; color:var(--muted);
                 border-radius:9px 9px 0 0; border-bottom:2.5px solid transparent; }
  .tabs button:hover { color:var(--text); background:var(--soft); }
  .tabs button.active { color:var(--accent-deep); border-bottom-color:var(--accent);
                        background:var(--accent-soft); }

  .wrap { display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:16px; padding:16px; }
  .wrap > * { min-width:0; }
  @media (max-width:920px) { .wrap { grid-template-columns:1fr; padding:12px; } }

  .card { background:var(--panel); border:1px solid var(--line); border-radius:12px;
          padding:15px; min-width:0; overflow:hidden; box-shadow:var(--shadow); }
  .card + .card { margin-top:16px; }
  .card h2 { margin:0 0 11px; font-size:11.5px; text-transform:uppercase;
             letter-spacing:.7px; color:var(--muted); font-weight:700; }

  .viewbox { position:relative; background:#eef2f8; border:1px solid var(--line);
             border-radius:10px; overflow:hidden; min-height:340px; max-height:68vh;
             display:flex; align-items:center; justify-content:center; }
  .view { max-width:100%; max-height:68vh; width:auto; height:auto;
          display:block; min-width:0; }
  .view:not([src]) { visibility:hidden; }
  .placeholder { position:absolute; color:var(--muted); font-size:13px; }

  label { display:block; margin:9px 0 3px; font-size:12px; color:var(--muted);
          font-weight:600; }
  input[type=text], input[type=number], input[type=email], select, textarea, input[type=file] {
    width:100%; padding:8px 10px; background:var(--panel); color:var(--text);
    border:1px solid var(--line); border-radius:8px; font:inherit; font-size:13px; }
  input:focus, select:focus, textarea:focus { outline:2px solid var(--accent-soft);
    border-color:var(--accent); }
  textarea { resize:vertical; min-height:54px; }
  .req::after { content:" *"; color:var(--bad); }
  .row { display:flex; gap:9px; flex-wrap:wrap; } .row > * { flex:1 1 130px; min-width:0; }

  .chips { display:flex; gap:7px; flex-wrap:wrap; margin-top:4px; }
  .chips label { display:flex; align-items:center; gap:6px; margin:0; padding:7px 11px;
                 background:var(--soft); border:1px solid var(--line); border-radius:8px;
                 cursor:pointer; color:var(--text); font-size:13px; font-weight:500; }
  .chips label:has(input:checked) { background:var(--accent-soft);
                                    border-color:var(--accent); color:var(--accent-deep); }
  .chips input { accent-color:var(--accent); }

  button.btn { width:100%; padding:10px; margin-top:11px; border:0; border-radius:9px;
               background:var(--accent); color:#fff; font-weight:700; cursor:pointer;
               font-size:13.5px; font-family:inherit; }
  button.btn:hover:not(:disabled) { background:var(--accent-deep); }
  button.ghost { background:var(--soft); color:var(--text); font-weight:600;
                 border:1px solid var(--line); }
  button.ghost:hover:not(:disabled) { background:var(--accent-soft); border-color:var(--accent); }
  button.btn:disabled { opacity:.5; cursor:not-allowed; }

  .grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
  @media (max-width:640px) { .grid { grid-template-columns:repeat(2,1fr); } }
  .stat { background:var(--soft); border:1px solid var(--line); border-radius:10px;
          padding:10px 12px; }
  .stat .k { font-size:11px; color:var(--muted); font-weight:600; }
  .stat .v { font-size:19px; font-weight:700; margin-top:2px; letter-spacing:-.3px; }

  table { width:100%; border-collapse:collapse; font-size:13px; }
  th { text-align:left; color:var(--muted); font-weight:700; font-size:10.5px;
       text-transform:uppercase; letter-spacing:.5px; padding:5px 7px; }
  td { padding:7px; border-top:1px solid var(--line); }
  tbody tr:hover { background:var(--soft); }

  .pill { padding:3px 10px; border-radius:99px; font-size:11px; font-weight:700; }
  .ok { background:var(--ok-soft); color:var(--ok); }
  .bad { background:var(--bad-soft); color:var(--bad); }
  .idle { background:var(--soft); color:var(--muted); }
  .muted { color:var(--muted); }
  .msg { margin-top:10px; font-size:12.5px; min-height:1.4em; color:var(--muted); }
  .msg.err { color:var(--bad); } .msg.good { color:var(--ok); }
  .hide { display:none !important; }

  .bar { height:9px; background:var(--soft); border:1px solid var(--line);
         border-radius:99px; overflow:hidden; margin-top:9px; }
  .bar i { display:block; height:100%; width:0; background:var(--accent);
           transition:width .25s ease; }
  .linkbtn { border:0; background:none; color:var(--bad); cursor:pointer;
             font:inherit; font-size:12px; font-weight:600; padding:2px 4px; }
  .linkbtn:hover { text-decoration:underline; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <h1>Face Track</h1>
    <span>detection and recognition &middot; FYP2 &middot; version 2</span>
    <span class="state" id="state"></span>
  </div>
  <nav class="tabs">
    <button data-tab="recognize" class="active">Recognize</button>
    <button data-tab="enroll">Add a face</button>
  </nav>
</header>

<!-- ===================================================== recognize -->
<div class="wrap" id="tab-recognize">
  <div>
    <div class="card">
      <h2>Preview</h2>
      <div class="viewbox">
        <div class="placeholder" id="phR">not running</div>
        <img class="view" id="viewR" alt="">
      </div>
      <div class="row" style="margin-top:9px">
        <button class="btn ghost" id="snap">Save snapshot</button>
        <button class="btn ghost" id="reset">Reset statistics</button>
        <button class="btn ghost" id="csv">Export CSV</button>
      </div>
    </div>

    <div class="card">
      <h2>Statistics</h2>
      <div class="grid" id="stats"></div>
      <table style="margin-top:10px">
        <thead><tr><th>Identified</th><th>ID</th><th>Frames</th><th>Share</th></tr></thead>
        <tbody id="people"><tr><td colspan="4" class="muted">nothing yet</td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="card" style="align-self:start">
    <h2>Source</h2>
    <div class="chips" id="srcKind">
      <label><input type="radio" name="src" value="camera" checked> Camera</label>
      <label><input type="radio" name="src" value="video"> Video</label>
      <label><input type="radio" name="src" value="photos"> Photos</label>
    </div>
    <div id="cameraBox">
      <label>Camera index</label>
      <input type="number" id="cameraIndex" value="0" min="0" max="8">
      <label>Rotation <span class="muted" style="font-weight:400">(if the camera is not upright)</span></label>
      <select id="rotate">
        <option value="0">none</option><option value="90">90&deg;</option>
        <option value="180">180&deg;</option><option value="270">270&deg;</option>
      </select>
    </div>
    <div id="fileBox" class="hide">
      <label>Upload a file to the server</label>
      <input type="file" id="upload">
      <label>Or a path already on the server</label>
      <select id="serverFile"></select>
      <label style="display:flex;align-items:center;gap:7px;margin-top:9px;font-weight:500">
        <input type="checkbox" id="loop" style="width:auto"> Loop when it ends
      </label>
    </div>

    <h2 style="margin-top:17px">Detection</h2>
    <div class="chips" id="detKind">
      <label><input type="radio" name="det" value="haar" checked> Haar</label>
      <label><input type="radio" name="det" value="yunet"> YuNet</label>
    </div>
    <div id="haarOpts">
      <div class="row">
        <div><label>scaleFactor</label><input type="number" id="scaleFactor" value="1.1" step="0.01" min="1.01"></div>
        <div><label>minNeighbors</label><input type="number" id="minNeighbors" value="5" min="1"></div>
      </div>
      <label>Minimum face size (px)</label>
      <input type="number" id="minSize" value="60" min="20" step="10">
    </div>
    <div id="yunetOpts" class="hide">
      <label>Score threshold</label>
      <input type="number" id="scoreThreshold" value="0.9" step="0.05" min="0.05" max="1">
    </div>

    <h2 style="margin-top:17px">Recognition</h2>
    <select id="recognizer"><option value="none">Off, detection only</option></select>
    <label>Threshold <span class="muted" style="font-weight:400">(blank uses config.py)</span></label>
    <input type="text" id="threshold" placeholder="default">

    <button class="btn" id="start">Start</button>
    <button class="btn ghost" id="stop" disabled>Stop</button>
    <div class="msg" id="msg"></div>
  </div>
</div>

<!-- ======================================================== enroll -->
<div class="wrap hide" id="tab-enroll">
  <div>
    <div class="card">
      <h2>Capture</h2>
      <div class="viewbox">
        <div class="placeholder" id="phE">press Start capture when ready</div>
        <img class="view" id="viewE" alt="">
      </div>
      <div class="bar"><i id="progbar"></i></div>
      <div class="msg" id="emsg"></div>
    </div>

    <div class="card">
      <h2>Enrolled people</h2>
      <table>
        <thead><tr><th>Name</th><th>ID</th><th>Programme</th><th>Section</th>
                   <th>Images</th><th>Enrolled</th><th></th></tr></thead>
        <tbody id="peopleList"><tr><td colspan="7" class="muted">nobody enrolled yet</td></tr></tbody>
      </table>
    </div>

    <div class="card">
      <h2>Train the recognizer</h2>
      <p class="muted" style="margin:0 0 4px;font-size:12.5px">
        A new person is only recognised after the model is rebuilt. Retrain here,
        then press Start on the Recognize tab to load it.</p>
      <div class="row">
        <div><label>Recognizer</label><select id="trainRec"></select></div>
        <div><label>Detector</label>
          <select id="trainDet"><option value="haar">Haar</option><option value="yunet">YuNet</option></select></div>
      </div>
      <button class="btn" id="train">Train now</button>
      <div class="msg" id="tmsg"></div>
      <table id="trainTable" class="hide" style="margin-top:8px">
        <thead><tr><th>Person</th><th>Images</th><th>Faces found</th><th>Rate</th></tr></thead>
        <tbody id="trainRows"></tbody>
      </table>
    </div>
  </div>

  <div class="card" style="align-self:start">
    <h2>Person details</h2>
    <label class="req">Student ID</label>
    <input type="text" id="f_student_id" placeholder="e.g. 210001">
    <label class="req">Full name</label>
    <input type="text" id="f_name" placeholder="e.g. Student Name">
    <label>Email</label>
    <input type="email" id="f_email" placeholder="e.g. 210001@student.kcst.edu.kw">
    <div class="row">
      <div><label>Programme</label><input type="text" id="f_programme" placeholder="e.g. Computer Engineering"></div>
      <div><label>Section</label><input type="text" id="f_section" placeholder="e.g. CE 492 - A"></div>
    </div>
    <label>Notes</label>
    <textarea id="f_notes" placeholder="anything worth recording"></textarea>

    <h2 style="margin-top:17px">Capture settings</h2>
    <div class="row">
      <div><label>Images to capture</label><input type="number" id="e_count" value="30" min="5" max="200"></div>
      <div><label>Camera index</label><input type="number" id="e_camera" value="0" min="0" max="8"></div>
    </div>
    <label>Rotation</label>
    <select id="e_rotate">
      <option value="0">none</option><option value="90">90&deg;</option>
      <option value="180">180&deg;</option><option value="270">270&deg;</option>
    </select>
    <label>Detector</label>
    <div class="chips" id="eDetKind">
      <label><input type="radio" name="edet" value="haar" checked> Haar</label>
      <label><input type="radio" name="edet" value="yunet"> YuNet</label>
    </div>
    <div class="row" style="margin-top:6px">
      <div><label>Sharpness floor</label><input type="number" id="e_blur" value="60" min="0" step="5"></div>
      <div><label>Frame gap</label><input type="number" id="e_gap" value="5" min="1"></div>
    </div>
    <label style="display:flex;align-items:center;gap:7px;margin-top:10px;font-weight:500">
      <input type="checkbox" id="e_replace" style="width:auto"> Replace this person's existing images
    </label>

    <button class="btn" id="estart">Start capture</button>
    <button class="btn ghost" id="estop" disabled>Stop</button>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const setMsg = (el, t, kind) => { const e = $(el); e.textContent = t || "";
  e.className = "msg" + (kind ? " " + kind : ""); };
let running = false, enrolling = false, activeTab = "recognize";

/* ------------------------------------------------------------- tabs */
function showTab(name) {
  activeTab = (name === "enroll") ? "enroll" : "recognize";
  document.querySelectorAll(".tabs button").forEach(x =>
    x.classList.toggle("active", x.dataset.tab === activeTab));
  $("tab-recognize").classList.toggle("hide", activeTab !== "recognize");
  $("tab-enroll").classList.toggle("hide", activeTab !== "enroll");
  syncStreams();
  if (activeTab === "enroll") loadPeople();
}
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
  history.replaceState(null, "", "#" + b.dataset.tab);
  showTab(b.dataset.tab);
});
// #enroll in the address opens that tab directly, so a tab can be bookmarked.
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));
showTab(location.hash.slice(1));

/* ---------------------------------------------------------- streams */
function syncStreams() {
  const live = running || enrolling;
  const wantR = activeTab === "recognize" && live;
  const wantE = activeTab === "enroll" && live;
  [["viewR", wantR, "phR"], ["viewE", wantE, "phE"]].forEach(([id, want, ph]) => {
    const img = $(id);
    if (want && !img.getAttribute("src")) { img.src = "stream.mjpg?t=" + Date.now(); $(ph).textContent = ""; }
    if (!want && img.getAttribute("src")) { img.removeAttribute("src"); }
  });
}
function clearStreams(label) {
  ["viewR", "viewE"].forEach(id => $(id).removeAttribute("src"));
  $("phR").textContent = label; $("phE").textContent = label;
}

/* --------------------------------------------------------- recognize */
function sourceKind() { return document.querySelector('input[name=src]:checked').value; }
function detKind()    { return document.querySelector('input[name=det]:checked').value; }

$("srcKind").onchange = () => {
  const cam = sourceKind() === "camera";
  $("cameraBox").classList.toggle("hide", !cam);
  $("fileBox").classList.toggle("hide", cam);
  loadFiles();
};
$("detKind").onchange = () => {
  const haar = detKind() === "haar";
  $("haarOpts").classList.toggle("hide", !haar);
  $("yunetOpts").classList.toggle("hide", haar);
};

async function loadConfig() {
  const cfg = await (await fetch("api/config")).json();
  const sel = $("recognizer"), tsel = $("trainRec");
  sel.innerHTML = '<option value="none">Off, detection only</option>';
  tsel.innerHTML = "";
  for (const r of cfg.recognizers) {
    const o = document.createElement("option");
    o.value = r.name; o.textContent = r.label + (r.trained ? "" : "  (not trained yet)");
    o.disabled = !r.trained; sel.appendChild(o);
    const t = document.createElement("option");
    t.value = r.name; t.textContent = r.label; tsel.appendChild(t);
  }
  if (!cfg.yunet_available) {
    document.querySelectorAll('input[value=yunet]').forEach(e => e.disabled = true);
    $("trainDet").querySelector('option[value=yunet]').disabled = true;
  }
  loadFiles();
}

async function loadFiles() {
  const files = await (await fetch("api/files")).json();
  const sel = $("serverFile");
  const list = sourceKind() === "video" ? files.videos : files.photos;
  sel.innerHTML = '<option value="">-- choose --</option>';
  for (const f of list) {
    const o = document.createElement("option");
    o.value = f.path; o.textContent = f.label; sel.appendChild(o);
  }
}

$("upload").onchange = async e => {
  const file = e.target.files[0];
  if (!file) return;
  setMsg("msg", "uploading " + file.name + " ...");
  const res = await fetch("api/upload?name=" + encodeURIComponent(file.name),
                          { method: "POST", body: file });
  const out = await res.json();
  if (!res.ok) { setMsg("msg", out.error || "upload failed", "err"); return; }
  setMsg("msg", "uploaded " + out.name, "good");
  await loadFiles(); $("serverFile").value = out.path;
};

$("start").onclick = async () => {
  const body = {
    detector: detKind(), recognizer: $("recognizer").value,
    threshold: $("threshold").value.trim() || null,
    scale_factor: parseFloat($("scaleFactor").value),
    min_neighbors: parseInt($("minNeighbors").value),
    min_size: parseInt($("minSize").value),
    score_threshold: parseFloat($("scoreThreshold").value),
    loop: $("loop").checked
  };
  body.rotate = parseInt($("rotate").value);
  if (sourceKind() === "camera") body.camera_index = parseInt($("cameraIndex").value);
  else {
    body.source = $("serverFile").value;
    if (!body.source) { setMsg("msg", "choose or upload a file first", "err"); return; }
  }
  setMsg("msg", "starting ...");
  const res = await fetch("api/start", { method:"POST", body: JSON.stringify(body) });
  const out = await res.json();
  if (!res.ok) { setMsg("msg", out.error || "could not start", "err"); return; }
  setMsg("msg", out.detector_description || "running");
  running = true; $("start").disabled = true; $("stop").disabled = false; syncStreams();
};

$("stop").onclick = async () => {
  await fetch("api/stop", { method:"POST" });
  running = false; $("start").disabled = false; $("stop").disabled = true;
  clearStreams("stopped"); setMsg("msg", "stopped, statistics kept");
};
$("reset").onclick = async () => { await fetch("api/reset", {method:"POST"}); setMsg("msg","statistics reset","good"); };
$("snap").onclick  = async () => {
  const out = await (await fetch("api/snapshot", {method:"POST"})).json();
  setMsg("msg", out.path ? "saved " + out.path : (out.error || "nothing to save"), out.path ? "good" : "err");
};
$("csv").onclick = () => { window.location = "api/export.csv"; };

/* ------------------------------------------------------------ enroll */
function details() {
  const d = {};
  for (const f of ["student_id","name","email","programme","section","notes"])
    d[f] = $("f_" + f).value.trim();
  return d;
}

$("estart").onclick = async () => {
  const d = details();
  if (!d.student_id || !d.name) { setMsg("emsg", "a student ID and a name are both required", "err"); return; }
  const body = Object.assign({}, d, {
    count: parseInt($("e_count").value),
    camera_index: parseInt($("e_camera").value),
    rotate: parseInt($("e_rotate").value),
    detector: document.querySelector('input[name=edet]:checked').value,
    blur_threshold: parseFloat($("e_blur").value),
    gap: parseInt($("e_gap").value),
    replace: $("e_replace").checked,
    score_threshold: parseFloat($("scoreThreshold").value)
  });
  setMsg("emsg", "starting capture ...");
  const res = await fetch("api/enroll/start", { method:"POST", body: JSON.stringify(body) });
  const out = await res.json();
  if (!res.ok) { setMsg("emsg", out.error || "could not start", "err"); return; }
  enrolling = true; running = false;
  $("estart").disabled = true; $("estop").disabled = false;
  $("start").disabled = true; $("stop").disabled = true;
  syncStreams();
};

$("estop").onclick = async () => {
  await fetch("api/enroll/stop", { method:"POST" });
  enrolling = false; $("estart").disabled = false; $("estop").disabled = true;
  $("start").disabled = false;
  clearStreams("stopped"); loadPeople();
};

async function loadPeople() {
  const people = await (await fetch("api/people")).json();
  const body = $("peopleList");
  if (!people.length) {
    body.innerHTML = "<tr><td colspan='7' class='muted'>nobody enrolled yet</td></tr>";
    return;
  }
  body.innerHTML = people.map(p =>
    "<tr><td><b>" + esc(p.name) + "</b></td><td class='muted'>" + esc(p.student_id || "-") +
    "</td><td class='muted'>" + esc(p.programme || "-") + "</td><td class='muted'>" +
    esc(p.section || "-") + "</td><td>" + p.images + "</td><td class='muted'>" +
    esc((p.enrolled_at || "").split(" ")[0] || "-") +
    "</td><td><button class='linkbtn' data-del='" + esc(p.folder) + "'>remove</button></td></tr>"
  ).join("");
  body.querySelectorAll("[data-del]").forEach(b => b.onclick = async () => {
    if (!confirm("Remove " + b.dataset.del + " and all their images?")) return;
    const r = await fetch("api/people/delete", { method:"POST",
      body: JSON.stringify({ folder: b.dataset.del }) });
    const o = await r.json();
    setMsg("emsg", r.ok ? "removed " + b.dataset.del : (o.error || "could not remove"),
           r.ok ? "good" : "err");
    loadPeople();
  });
}
const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

$("train").onclick = async () => {
  setMsg("tmsg", "training ...");
  $("train").disabled = true;
  const res = await fetch("api/train", { method:"POST", body: JSON.stringify({
    recognizer: $("trainRec").value, detector: $("trainDet").value,
    score_threshold: parseFloat($("scoreThreshold").value) })});
  const out = await res.json();
  if (!res.ok) { setMsg("tmsg", out.error || "could not start training", "err");
                 $("train").disabled = false; return; }
  pollTraining();
};

async function pollTraining() {
  const s = await (await fetch("api/train/status")).json();
  if (s.state === "running") { setMsg("tmsg", s.message || "training ..."); setTimeout(pollTraining, 600); return; }
  $("train").disabled = false;
  setMsg("tmsg", s.message, s.state === "error" ? "err" : "good");
  if (s.progress && s.progress.length) {
    $("trainTable").classList.remove("hide");
    $("trainRows").innerHTML = s.progress.map(p =>
      "<tr><td>" + esc(p.name) + "</td><td>" + p.images + "</td><td>" + p.faces_found +
      "</td><td class='" + (p.rate >= 80 ? "muted" : "bad") + "'>" + p.rate + "%</td></tr>").join("");
  }
  if (s.state === "done") loadConfig();
}

/* -------------------------------------------------------- statistics */
const FIELDS = [
  ["frames","Frames"], ["total_faces","Faces"], ["detection_rate","Detection rate"],
  ["throughput_fps","Throughput"], ["mean_detect_ms","Detection"],
  ["mean_pipeline_ms","Pipeline mean"], ["max_pipeline_ms","Pipeline max"],
  ["unknown","Unknown faces"]
];
const UNITS = { detection_rate:"%", throughput_fps:" fps", mean_detect_ms:" ms",
                mean_pipeline_ms:" ms", max_pipeline_ms:" ms" };

async function poll() {
  try {
    const s = await (await fetch("api/status")).json();
    const e = s.enroll || {};
    const live = s.state === "running" || e.state === "running";
    $("state").innerHTML = live
      ? '<span class="pill ok">' + (e.state === "running" ? "capturing" : "running") + '</span>'
      : '<span class="pill ' + (s.state === "error" ? "bad" : "idle") + '">' + s.state + '</span>';

    if (s.state === "running" && !running && !enrolling) {
      running = true; $("start").disabled = true; $("stop").disabled = false; syncStreams();
    }
    if (s.state !== "running" && running) {
      running = false; $("start").disabled = false; $("stop").disabled = true;
      clearStreams(s.state === "error" ? "error" : "finished");
      if (s.state === "error") setMsg("msg", s.error, "err");
      else if (s.state === "finished") setMsg("msg", "source finished");
    }

    if (e.state === "running") {
      const pct = e.target ? Math.round(100 * e.saved / e.target) : 0;
      $("progbar").style.width = pct + "%";
      setMsg("emsg", e.saved + " / " + e.target + " captured - " + (e.message || ""));
    } else if (enrolling) {
      enrolling = false;
      $("estart").disabled = false; $("estop").disabled = true; $("start").disabled = false;
      $("progbar").style.width = (e.target ? Math.round(100 * e.saved / e.target) : 0) + "%";
      clearStreams(e.state === "error" ? "error" : "capture finished");
      setMsg("emsg", e.state === "error" ? (e.error || "capture failed")
             : "captured " + e.saved + " images into " + e.folder + ". Train the recognizer below.",
             e.state === "error" ? "err" : "good");
      loadPeople();
    }

    const st = s.stats;
    $("stats").innerHTML = FIELDS.map(([k, label]) =>
      '<div class="stat"><div class="k">' + label + '</div><div class="v">' +
      st[k] + (UNITS[k] || "") + '</div></div>').join("");

    const rows = st.people.map(p =>
      "<tr><td>" + esc(p.name) + "</td><td class='muted'>" + esc(p.id || "-") +
      "</td><td>" + p.frames + "</td><td class='muted'>" + p.share + "%</td></tr>").join("");
    const unknown = st.unknown
      ? "<tr><td class='muted'>Unknown</td><td class='muted'>-</td><td>" + st.unknown + "</td><td class='muted'>-</td></tr>" : "";
    $("people").innerHTML = (rows + unknown) || "<tr><td colspan='4' class='muted'>nothing yet</td></tr>";
  } catch (err) { /* server busy, try again next tick */ }
  setTimeout(poll, 700);
}

loadConfig(); loadPeople(); poll();
</script>
</body>
</html>
"""


# ==================================================================== handler
class Handler(BaseHTTPRequestHandler):
    server_version = "FaceTrack/0.2"
    session: PipelineSession = None          # set by serve()
    enroll: EnrollmentSession = None
    training: TrainingJob = None

    def log_message(self, fmt, *args):       # keep the console readable
        if self.path.startswith(("/stream", "/api/status")):
            return
        super().log_message(fmt, *args)

    # ------------------------------------------------------------- helpers
    def _json(self, payload, status=200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    @classmethod
    def _active(cls):
        """Whichever of the two is producing frames right now."""
        if cls.enroll is not None and cls.enroll.state == "running":
            return cls.enroll
        return cls.session

    @staticmethod
    def _detector_from(body: dict):
        name = body.get("detector", "haar")
        if name == "haar":
            return get_detector(
                "haar",
                scale_factor=float(body.get("scale_factor", config.HAAR_SCALE_FACTOR)),
                min_neighbors=int(body.get("min_neighbors", config.HAAR_MIN_NEIGHBORS)),
                min_size=(int(body.get("min_size", config.HAAR_MIN_SIZE[0])),) * 2)
        return get_detector("yunet", score_threshold=float(
            body.get("score_threshold", config.YUNET_SCORE_THRESHOLD)))

    # ------------------------------------------------------------- routing
    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if path == "/":
                self._send_page()
            elif path == "/stream.mjpg":
                self._send_stream()
            elif path == "/api/status":
                payload = self.session.status()
                payload["enroll"] = self.enroll.status()
                self._json(payload)
            elif path == "/api/config":
                self._json(self._config_payload())
            elif path == "/api/files":
                self._json(self._files_payload())
            elif path == "/api/people":
                self._json(list_people())
            elif path == "/api/train/status":
                self._json(self.training.status())
            elif path == "/api/export.csv":
                self._send_csv()
            else:
                self._json({"error": "not found"}, 404)
        except ConnectionError:
            pass                                  # the browser navigated away

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if path == "/api/start":
                self._start(self._read_json())
            elif path == "/api/stop":
                self.session.stop()
                self._json({"ok": True})
            elif path == "/api/reset":
                self.session.stats.reset()
                self._json({"ok": True})
            elif path == "/api/snapshot":
                saved = self._active().save_snapshot()
                self._json({"path": str(saved)} if saved else {"error": "nothing to save yet"})
            elif path == "/api/upload":
                self._upload(parse_qs(urlparse(self.path).query))
            elif path == "/api/enroll/start":
                self._enroll_start(self._read_json())
            elif path == "/api/enroll/stop":
                self.enroll.stop()
                self._json({"ok": True})
            elif path == "/api/people/delete":
                folder = (self._read_json().get("folder") or "").strip()
                ok = delete_person(folder) if folder else False
                self._json({"ok": True} if ok else {"error": "no such person"},
                           200 if ok else 400)
            elif path == "/api/train":
                self._train(self._read_json())
            else:
                self._json({"error": "not found"}, 404)
        except ConnectionError:
            pass

    # --------------------------------------------------------------- pages
    def _send_page(self) -> None:
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()

        last_id = -1
        idle = 0
        while True:
            source = self._active()
            if source.state != "running":
                idle += 1
                if idle > 40:                       # about four seconds
                    break
            current = source.frame_id
            if current == last_id:
                time.sleep(0.02)
                continue
            last_id = current
            idle = 0
            jpeg = source.latest_jpeg()
            if jpeg is None:
                time.sleep(0.05)
                continue
            self.wfile.write(f"--{BOUNDARY}\r\n".encode())
            self.wfile.write(b"Content-Type: image/jpeg\r\n")
            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
            self.wfile.write(jpeg)
            self.wfile.write(b"\r\n")

    def _send_csv(self) -> None:
        status = self.session.status()
        rows = self.session.stats.export_rows(
            status["detector"], status["recognizer"], status["source"])
        buffer = io.StringIO()
        if rows:
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        body = buffer.getvalue().encode("utf-8")
        name = f"facetrack_session_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.send_response(200)
        self.send_header("Content-Type", "text/csv")
        self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------- payloads
    @staticmethod
    def _config_payload() -> dict:
        recognizers = []
        for name in RECOGNIZER_NAMES:
            recognizers.append({
                "name": name,
                "label": {"lbph": "LBPH (baseline)",
                          "sface": "SFace (embeddings)"}.get(name, name),
                "trained": Path(default_model_path(name)).is_file(),
            })
        return {
            "detectors": list(DETECTOR_NAMES),
            "recognizers": recognizers,
            "yunet_available": config.YUNET_MODEL.is_file(),
            "detail_fields": list(DETAIL_FIELDS),
        }

    @staticmethod
    def _files_payload() -> dict:
        videos, photos = [], []
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        for path in sorted(UPLOAD_DIR.iterdir()):
            if not path.is_file():
                continue
            entry = {"path": str(path), "label": f"uploads / {path.name}"}
            if path.suffix.lower() in config.IMAGE_EXTENSIONS:
                photos.append(entry)
            else:
                videos.append(entry)

        # Enrolled people are the handiest photo folders to replay.
        if config.DATASET_DIR.is_dir():
            for folder in sorted(p for p in config.DATASET_DIR.iterdir() if p.is_dir()):
                if any(f.suffix.lower() in config.IMAGE_EXTENSIONS for f in folder.iterdir()):
                    photos.append({"path": str(folder), "label": f"dataset / {folder.name}"})

        return {"videos": videos, "photos": photos}

    # -------------------------------------------------------------- actions
    def _start(self, body: dict) -> None:
        if self.enroll.state == "running":
            self._json({"error": "an enrollment capture is using the camera"}, 409)
            return
        threshold = body.get("threshold")
        try:
            threshold = float(threshold) if threshold not in (None, "") else None
        except (TypeError, ValueError):
            self._json({"error": "threshold must be a number"}, 400)
            return

        self.session.stop()
        try:
            self.session.start(
                source=body.get("source") or None,
                camera_index=int(body.get("camera_index", config.CAMERA_INDEX)),
                detector=body.get("detector", "haar"),
                recognizer=body.get("recognizer", "none"),
                threshold=threshold,
                loop=bool(body.get("loop")),
                rotate=int(body.get("rotate", config.CAMERA_ROTATION)),
                scale_factor=body.get("scale_factor", config.HAAR_SCALE_FACTOR),
                min_neighbors=body.get("min_neighbors", config.HAAR_MIN_NEIGHBORS),
                min_size=body.get("min_size", config.HAAR_MIN_SIZE[0]),
                score_threshold=body.get("score_threshold", config.YUNET_SCORE_THRESHOLD),
            )
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)
            return
        self._json({"ok": True, "detector_description": self.session.detector_description})

    def _enroll_start(self, body: dict) -> None:
        # Only one thing can hold the camera, so recognition stands down first.
        self.session.stop()
        self.enroll.stop()
        details = {field: body.get(field, "") for field in DETAIL_FIELDS}
        try:
            self.enroll.start(
                details,
                count=int(body.get("count", config.CAPTURE_DEFAULT_COUNT)),
                detector=body.get("detector", "haar"),
                camera_index=int(body.get("camera_index", config.CAMERA_INDEX)),
                source=body.get("source") or None,
                blur_threshold=float(body.get("blur_threshold", config.CAPTURE_BLUR_THRESHOLD)),
                gap=int(body.get("gap", config.CAPTURE_MIN_FRAME_GAP)),
                replace=bool(body.get("replace")),
                rotate=int(body.get("rotate", config.CAMERA_ROTATION)),
                score_threshold=body.get("score_threshold", config.YUNET_SCORE_THRESHOLD),
            )
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)
            return
        self._json({"ok": True, "folder": self.enroll.status()["folder"]})

    def _train(self, body: dict) -> None:
        if self.training.running:
            self._json({"error": "training is already running"}, 409)
            return
        name = body.get("recognizer", "lbph")
        if name not in RECOGNIZER_NAMES:
            self._json({"error": f"unknown recognizer {name}"}, 400)
            return
        try:
            detector = self._detector_from(body)
            self.training.start(name, detector)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)
            return
        self._json({"ok": True})

    def _upload(self, query: dict) -> None:
        name = (query.get("name") or ["upload.bin"])[0]
        safe = Path(name).name.replace("\\", "_")
        if not safe:
            self._json({"error": "bad file name"}, 400)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._json({"error": "empty upload"}, 400)
            return
        if length > MAX_UPLOAD:
            self._json({"error": f"file larger than {MAX_UPLOAD // (1024 * 1024)} MB"}, 413)
            return

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = UPLOAD_DIR / safe
        remaining = length
        with open(target, "wb") as handle:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                handle.write(chunk)
                remaining -= len(chunk)
        self._json({"ok": True, "name": safe, "path": str(target)})


# ====================================================================== serve
def local_ip() -> str:
    """Best guess at the address other devices on the Wi-Fi should use."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))       # no traffic is sent, this just picks a route
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def serve(host: str = "0.0.0.0", port: int = 8000,
          session: Optional[PipelineSession] = None) -> int:
    config.ensure_dirs()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Handler.session = session or PipelineSession()
    Handler.enroll = EnrollmentSession()
    Handler.training = TrainingJob()

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True

    shown = local_ip() if host in ("0.0.0.0", "") else host
    print("=" * 62)
    print("Face Track web interface (version 2)")
    print("=" * 62)
    print(f"  on this machine : http://localhost:{port}/")
    if host in ("0.0.0.0", ""):
        print(f"  on the Wi-Fi    : http://{shown}:{port}/")
        print("  (open that address on a laptop or phone on the same network)")
    print("\npress Ctrl+C to stop\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        Handler.session.stop()
        Handler.enroll.stop()
        httpd.server_close()
    return 0
