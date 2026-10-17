"use strict";
const $ = (sel) => document.querySelector(sel);
const newDialog = $("#new-dialog"), itemDialog = $("#item-dialog"), voiceDialog = $("#voice-dialog");
const player = $("#player");
const LANG_NAMES = { en: "English", yue: "粵語", zh: "普通話" };
let jobs = [], voices = [], defaultVoice = "default", currentItem = null, playingId = null;
let source = "url", pollTimer = null;

// ---- helpers -------------------------------------------------------------------------------
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts = {}) {
  const init = { credentials: "same-origin", ...opts, headers: { ...(opts.headers || {}) } };
  if (opts.json !== undefined) {
    init.body = JSON.stringify(opts.json);
    init.headers["Content-Type"] = "application/json";
  }
  const r = await fetch(path, init);
  if (r.status === 401 && path !== "/v1/login") { showLogin(); throw new Error("Please sign in"); }
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const d = await r.json();
      msg = typeof d.detail === "string" ? d.detail : (d.detail || []).map((e) => e.msg.replace(/^Value error, /, "")).join("; ") || msg;
    } catch {}
    throw new Error(msg);
  }
  return r.status === 204 ? null : r.json();
}

function toast(msg, ms = 2500) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), ms);
}

function fmtDuration(sec) {
  if (sec == null) return "";
  sec = Math.round(sec);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
}

function fmtDate(ts) {
  const d = new Date(ts * 1000), now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  return sameDay ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                 : d.toLocaleDateString([], { year: d.getFullYear() === now.getFullYear() ? undefined : "numeric", month: "short", day: "numeric" });
}

const host = (url) => { try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return ""; } };
const audioUrl = (job, download = false) => `/v1/jobs/${job.id}/audio${download ? "?download=true" : ""}`;

// ---- auth ----------------------------------------------------------------------------------
function showLogin() {
  $("#app").hidden = true;
  $("#login").hidden = false;
  $("#login-key").focus();
}

$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#login-error").hidden = true;
  try {
    await api("/v1/login", { method: "POST", json: { api_key: $("#login-key").value } });
    $("#login-key").value = "";
    start();
  } catch (err) {
    $("#login-error").textContent = err.message;
    $("#login-error").hidden = false;
  }
});

$("#logout-btn").addEventListener("click", async () => {
  await api("/v1/logout", { method: "POST" });
  location.reload();
});

// ---- tabs ----------------------------------------------------------------------------------
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  $("#tab-library").hidden = b.dataset.tab !== "library";
  $("#tab-voices").hidden = b.dataset.tab !== "voices";
  $("#search").hidden = b.dataset.tab !== "library";
  if (b.dataset.tab === "voices") renderVoices();
}));

// ---- library -------------------------------------------------------------------------------
async function loadJobs() {
  const d = await api("/v1/jobs?limit=500");
  jobs = d.jobs;
  renderLibrary();
  schedulePoll();
}

function schedulePoll() {
  clearTimeout(pollTimer);
  const active = jobs.some((j) => j.status === "queued" || j.status === "running");
  pollTimer = setTimeout(() => loadJobs().catch(() => {}), active ? 3000 : 30000);
}

function renderLibrary() {
  const q = $("#search").value.trim().toLowerCase();
  const shown = q ? jobs.filter((j) => j.title.toLowerCase().includes(q) || (j.source_url || "").toLowerCase().includes(q)) : jobs;
  $("#library-empty").hidden = jobs.length > 0;
  $("#library").innerHTML = shown.map((j) => {
    const done = j.status === "done";
    const active = j.status === "queued" || j.status === "running";
    const meta = [
      `<span>${esc(fmtDate(j.created_at))}</span>`,
      done ? `<span>${fmtDuration(j.duration_seconds)}</span>` : "",
      j.source_url ? `<span>${esc(host(j.source_url))}</span>` : "",
      ...(j.languages || []).map((l) => `<span class="chip">${esc(LANG_NAMES[l] || l)}</span>`),
      done ? "" : `<span class="chip status-${esc(j.status)}">${esc(j.status === "running" ? `${Math.round(j.progress * 100)}%` : j.status)}</span>`,
      j.share_url ? `<span class="chip shared">shared</span>` : "",
    ].join("");
    return `<div class="item ${playingId === j.id ? "playing" : ""}" data-id="${esc(j.id)}">
      <button class="thumb" data-play="${esc(j.id)}" ${done ? "" : "disabled"} title="Play">${done ? (playingId === j.id && !player.paused ? "❚❚" : "▶") : active ? "⋯" : "!"}</button>
      <div>
        <div class="item-title">${esc(j.title || "(untitled)")}</div>
        <div class="item-meta">${meta}</div>
        ${active ? `<div class="progress"><div style="width:${Math.max(3, j.progress * 100)}%"></div></div>` : ""}
        ${j.status === "failed" ? `<div class="error small">${esc(j.error)}</div>` : ""}
      </div>
      <div class="item-actions">
        ${done ? `<a class="button ghost" href="${audioUrl(j, true)}" title="Download" onclick="event.stopPropagation()">⬇</a>` : ""}
      </div>
    </div>`;
  }).join("");
}

$("#library").addEventListener("click", (e) => {
  const play = e.target.closest("[data-play]");
  if (play) { e.stopPropagation(); togglePlay(play.dataset.play); return; }
  const item = e.target.closest(".item");
  if (item) openItem(item.dataset.id);
});
$("#search").addEventListener("input", renderLibrary);

// ---- player --------------------------------------------------------------------------------
function togglePlay(id) {
  if (playingId === id) { player.paused ? player.play() : player.pause(); return; }
  const j = jobs.find((x) => x.id === id);
  if (!j) return;
  playingId = id;
  player.src = audioUrl(j);
  $("#player-title").textContent = j.title;
  $("#player-sub").textContent = [fmtDuration(j.duration_seconds), host(j.source_url)].filter(Boolean).join(" · ");
  $("#player-bar").hidden = false;
  player.play();
  renderLibrary();
}
function closePlayer() {
  player.pause();
  player.removeAttribute("src");
  playingId = null;
  $("#player-bar").hidden = true;
  renderLibrary();
}
player.addEventListener("play", renderLibrary);
player.addEventListener("pause", renderLibrary);

// ---- item dialog ---------------------------------------------------------------------------
async function openItem(id) {
  const j = jobs.find((x) => x.id === id);
  if (!j) return;
  currentItem = j;
  $("#item-title").value = j.title;
  const meta = [fmtDate(j.created_at), fmtDuration(j.duration_seconds), `${j.characters} characters`,
                `voice: ${j.voice}`, `speed ${j.speed}×`, j.format.toUpperCase()].filter(Boolean);
  $("#item-meta").innerHTML = meta.map(esc).join(" · ") +
    (j.source_url ? ` · <a href="${esc(j.source_url)}" target="_blank" rel="noopener noreferrer">source</a>` : "");
  const done = j.status === "done";
  $("#item-play").disabled = !done;
  $("#item-download").hidden = !done;
  $("#item-download").href = done ? audioUrl(j, true) : "#";
  $("#item-share").hidden = !done;
  $("#item-delete").disabled = j.status === "running";
  renderShare();
  $("#item-text").textContent = "Loading…";
  itemDialog.showModal();
  fitTitle();
  try { $("#item-text").textContent = (await api(`/v1/jobs/${id}/text`)).text || "(not stored)"; }
  catch (e) { $("#item-text").textContent = e.message; }
}

function renderShare() {
  const url = currentItem.share_url ? location.origin + currentItem.share_url : "";
  $("#item-share-box").hidden = !url;
  $("#item-share-url").value = url;
  $("#item-share").textContent = url ? "Shared ✓" : "Share";
}

async function patchItem(body) {
  const updated = await api(`/v1/jobs/${currentItem.id}`, { method: "PATCH", json: body });
  Object.assign(currentItem, updated);
  renderLibrary();
}

function fitTitle() {
  const el = $("#item-title");
  el.style.height = "auto";
  el.style.height = `${el.scrollHeight + 2}px`;
}
$("#item-title").addEventListener("input", fitTitle);
$("#item-title").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); e.target.blur(); } });
$("#item-title").addEventListener("change", async () => {
  try { await patchItem({ title: $("#item-title").value }); toast("Renamed"); } catch (e) { toast(e.message); }
});
$("#item-play").addEventListener("click", () => { itemDialog.close(); togglePlay(currentItem.id); });
$("#item-share").addEventListener("click", async () => {
  if (!currentItem.share_url) { await patchItem({ shared: true }); renderShare(); }
  copyShare();
});
$("#item-copy").addEventListener("click", copyShare);
$("#item-unshare").addEventListener("click", async () => { await patchItem({ shared: false }); renderShare(); toast("Link revoked"); });
async function copyShare() {
  $("#item-share-url").select();
  try { await navigator.clipboard.writeText($("#item-share-url").value); toast("Share link copied"); }
  catch { document.execCommand("copy"); toast("Share link copied"); }
}
$("#item-delete").addEventListener("click", async () => {
  if (!confirm(`Delete "${currentItem.title}"? This cannot be undone.`)) return;
  try {
    await api(`/v1/jobs/${currentItem.id}`, { method: "DELETE" });
    if (playingId === currentItem.id) closePlayer();
    itemDialog.close();
    toast("Deleted");
    loadJobs();
  } catch (e) { toast(e.message); }
});

// ---- new conversion ------------------------------------------------------------------------
function openNew() {
  $("#new-error").hidden = true;
  $("#new-voice").innerHTML = voices.map((v) =>
    `<option value="${esc(v.name)}" ${v.name === defaultVoice ? "selected" : ""}>${esc(v.name)} (${esc(LANG_NAMES[v.language] || v.language)})</option>`).join("");
  setSource(source);
  newDialog.showModal();
}
$("#new-btn").addEventListener("click", openNew);

function setSource(s) {
  source = s;
  document.querySelectorAll(".seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.src === s));
  $("#src-url").hidden = s !== "url";
  $("#text-label").hidden = s === "url" && !$("#new-text").value;
}
document.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => setSource(b.dataset.src)));
$("#new-speed").addEventListener("input", () => ($("#speed-val").textContent = `${(+$("#new-speed").value).toFixed(1)}×`));
$("#new-text").addEventListener("input", () => ($("#char-count").textContent = `${$("#new-text").value.length} characters`));

function newError(msg) { $("#new-error").textContent = msg; $("#new-error").hidden = !msg; }

$("#fetch-btn").addEventListener("click", async () => {
  const url = $("#new-url").value.trim();
  if (!url) return newError("Enter a URL");
  newError("");
  $("#fetch-btn").disabled = true;
  $("#fetch-btn").textContent = "Fetching…";
  try {
    const a = await api("/v1/fetch", { method: "POST", json: { url } });
    if (!$("#new-title").value) $("#new-title").value = a.title;
    $("#new-text").value = a.text;
    $("#new-text").dispatchEvent(new Event("input"));
    $("#text-label").hidden = false;
  } catch (e) { newError(e.message); }
  finally { $("#fetch-btn").disabled = false; $("#fetch-btn").textContent = "Preview"; }
});

$("#convert-btn").addEventListener("click", async () => {
  newError("");
  const body = {
    title: $("#new-title").value.trim(),
    voice: $("#new-voice").value,
    language: $("#new-lang").value,
    speed: +$("#new-speed").value,
    format: $("#new-format").value,
  };
  const text = $("#new-text").value.trim(), url = $("#new-url").value.trim();
  if (text) {
    body.text = text;  // previewed (and possibly edited) article text wins over re-fetching
    if (source === "url" && url) body.source_url = url;
  } else if (source === "url" && url) body.url = url;
  else return newError(source === "url" ? "Enter a URL" : "Paste some text");

  $("#convert-btn").disabled = true;
  $("#convert-btn").textContent = "Starting…";
  try {
    const job = await api("/v1/jobs", { method: "POST", json: body });
    newDialog.close();
    $("#new-form").reset();
    $("#new-text").value = "";
    $("#speed-val").textContent = "1.0×";
    $("#char-count").textContent = "";
    toast(`Converting "${job.title}"…`);
    loadJobs();
  } catch (e) { newError(e.message); }
  finally { $("#convert-btn").disabled = false; $("#convert-btn").textContent = "Convert"; }
});

// ---- voices --------------------------------------------------------------------------------
async function loadVoices() {
  const d = await api("/v1/voices");
  voices = d.voices;
  defaultVoice = d.default;
}

function renderVoices() {
  $("#voices").innerHTML = voices.map((v) => `
    <div class="card voice">
      <div class="voice-name">${esc(v.name)} <span class="chip">${esc(LANG_NAMES[v.language] || v.language)}</span>
        ${v.name === defaultVoice ? '<span class="chip shared">default</span>' : ""}</div>
      ${v.description ? `<div class="muted small">${esc(v.description)}</div>` : ""}
      <div class="small">“${esc(v.transcript)}”</div>
      <audio controls preload="none" src="/v1/voices/${encodeURIComponent(v.name)}/audio"></audio>
      ${v.name === defaultVoice ? "" : `<div><button class="danger" data-del-voice="${esc(v.name)}">Delete</button></div>`}
    </div>`).join("");
}

$("#voices").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-del-voice]");
  if (!b || !confirm(`Delete voice "${b.dataset.delVoice}"?`)) return;
  try {
    await api(`/v1/voices/${encodeURIComponent(b.dataset.delVoice)}`, { method: "DELETE" });
    await loadVoices();
    renderVoices();
    toast("Voice deleted");
  } catch (err) { toast(err.message); }
});

function openVoiceDialog() { $("#voice-error").hidden = true; voiceDialog.showModal(); }

$("#voice-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#voice-error").hidden = true;
  try {
    await api("/v1/voices", { method: "POST", body: new FormData(e.target) });
    voiceDialog.close();
    e.target.reset();
    await loadVoices();
    renderVoices();
    toast("Voice added");
  } catch (err) {
    $("#voice-error").textContent = err.message;
    $("#voice-error").hidden = false;
  }
});

// ---- start ---------------------------------------------------------------------------------
async function start() {
  const s = await fetch("/v1/session", { credentials: "same-origin" }).then((r) => r.json());
  if (!s.authenticated) return showLogin();
  $("#login").hidden = true;
  $("#app").hidden = false;
  $("#logout-btn").hidden = !s.auth_required;
  await Promise.all([loadVoices(), loadJobs()]);
}
start();
