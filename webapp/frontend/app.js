(() => {
  const API = location.protocol.startsWith("http") ? "" : "http://localhost:8000";
  const $ = (id) => document.getElementById(id);
  const S = { view: "describe", blob: null, imgSize: null, last: null, detail: "rich", stream: null,
              liveTimer: null, liveBusy: false, lastLiveCaption: "", boxes: true, history: [] };
  const LIVE_EVERY_MS = 4000;

  /* ---------- helpers ---------- */
  const toast = (msg) => { const t = $("toast"); t.textContent = msg; t.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), 2600); };
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const b64Blob = (b64, mime) => { const bin = atob(b64); const a = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i); return new Blob([a], { type: mime }); };
  const busy = (on, text) => { $("busy").hidden = !on; if (text) $("busyText").textContent = text; };
  const similar = (a, b) => { const A = new Set(a.toLowerCase().split(/\W+/)), B = new Set(b.toLowerCase().split(/\W+/)); const i = [...A].filter((x) => B.has(x)).length; return i / Math.max(1, new Set([...A, ...B]).size); };
  const settings = () => ({ lang: $("lang").value, accent: $("accent").value, detail: S.detail });

  /* ---------- health ---------- */
  fetch(`${API}/api/health`).then((r) => r.json()).then((d) => {
    $("status").classList.add("ok"); $("statusText").textContent = `Models ready · ${d.device}`;
  }).catch(() => { $("status").classList.add("bad"); $("statusText").textContent = "Server offline"; });

  /* ---------- views ---------- */
  const VIEW_TEXT = { describe: ["Describe and speak", "What I see"], read: ["Read text aloud", "Text in this image"], live: ["Start live narration", "Live narration"] };
  function setView(v) {
    if (S.view === "live" && v !== "live") stopLive();
    S.view = v;
    document.querySelectorAll("[data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
    const isHist = v === "history";
    $("workView").hidden = isHist; $("historyView").hidden = !isHist; $("recentWrap").hidden = isHist;
    if (isHist) return renderHistory();
    $("goBtn").textContent = VIEW_TEXT[v][0]; $("resultLabel").textContent = VIEW_TEXT[v][1];
    $("goBtn").disabled = v === "live" ? false : !S.blob;
    if (v === "live") startCamera();
  }
  document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));

  /* ---------- image input ---------- */
  function setImage(blob) {
    stopCamera();
    S.blob = blob; S.last = null;
    const url = URL.createObjectURL(blob);
    const img = $("preview");
    img.onload = () => { S.imgSize = [img.naturalWidth, img.naturalHeight]; layoutOverlay(); };
    img.src = url; img.hidden = false;
    $("empty").hidden = true; $("stageBar").hidden = false; $("overlay").innerHTML = "";
    $("goBtn").disabled = S.view === "history";
    $("qaList").innerHTML = "";
  }
  $("chooseBtn").onclick = $("replaceBtn").onclick = (e) => { e.stopPropagation(); $("file").click(); };
  $("file").onchange = (e) => { const f = e.target.files[0]; if (f) setImage(f); e.target.value = ""; };
  $("stage").addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && e.target === $("stage")) { e.preventDefault(); $("file").click(); } });
  ["dragenter", "dragover"].forEach((ev) => $("stage").addEventListener(ev, (e) => { e.preventDefault(); $("stage").classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => $("stage").addEventListener(ev, (e) => { e.preventDefault(); $("stage").classList.remove("drag"); }));
  $("stage").addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f && f.type.startsWith("image/")) setImage(f); else toast("Drop an image file."); });
  document.addEventListener("paste", (e) => { const it = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/")); if (it) setImage(it.getAsFile()); });

  /* ---------- camera ---------- */
  async function startCamera() {
    if (S.stream) return;
    try {
      S.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false });
      const v = $("video"); v.srcObject = S.stream; v.hidden = false;
      v.onloadedmetadata = () => { S.imgSize = [v.videoWidth, v.videoHeight]; layoutOverlay(); };
      $("preview").hidden = true; $("empty").hidden = true; $("stageBar").hidden = false;
      $("snapBtn").hidden = S.view === "live"; $("camBtn").textContent = "Close camera"; $("overlay").innerHTML = "";
    } catch (err) { toast("Camera isn't available. Check browser permissions."); if (S.view === "live") setView("describe"); }
  }
  function stopCamera() {
    if (!S.stream) return;
    S.stream.getTracks().forEach((t) => t.stop()); S.stream = null;
    $("video").hidden = true; $("snapBtn").hidden = true; $("camBtn").textContent = "Use camera";
    if (S.blob) { $("preview").hidden = false; S.imgSize = [$("preview").naturalWidth, $("preview").naturalHeight]; }
    else { $("empty").hidden = false; $("stageBar").hidden = true; }
  }
  const grabFrame = () => new Promise((res) => {
    const v = $("video"), c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight; c.getContext("2d").drawImage(v, 0, 0);
    c.toBlob(res, "image/jpeg", 0.88);
  });
  $("camBtn").onclick = $("camBtn2").onclick = (e) => { e.stopPropagation(); S.stream ? stopCamera() : startCamera(); };
  $("snapBtn").onclick = async () => { const b = await grabFrame(); setImage(b); if (S.view !== "history") analyze(); };

  /* ---------- overlay boxes ---------- */
  function layoutOverlay() {
    if (!S.imgSize) return;
    const st = $("stage").getBoundingClientRect(), [nw, nh] = S.imgSize;
    const k = Math.min(st.width / nw, st.height / nh), w = nw * k, h = nh * k;
    Object.assign($("overlay").style, { left: `${(st.width - w) / 2}px`, top: `${(st.height - h) / 2}px`, width: `${w}px`, height: `${h}px` });
  }
  window.addEventListener("resize", layoutOverlay);
  function drawBoxes(dets = [], ocr = []) {
    const ov = $("overlay"); ov.innerHTML = "";
    const add = ([x1, y1, x2, y2], label, cls) => {
      const d = document.createElement("div"); d.className = `box ${cls}${y1 < 0.08 ? " below" : ""}`;
      Object.assign(d.style, { left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%` });
      const s = document.createElement("span"); s.textContent = label; d.appendChild(s); ov.appendChild(d);
    };
    dets.slice(0, 12).forEach((o) => add(o.box, `${o.label} · ${Math.round(o.conf * 100)}%`, ""));
    ocr.slice(0, 6).forEach((o) => add(o.box, "text", "text"));
    layoutOverlay();
  }
  $("boxesBtn").onclick = () => { S.boxes = !S.boxes; $("overlay").classList.toggle("hide", !S.boxes); $("boxesBtn").textContent = S.boxes ? "Hide boxes" : "Show boxes"; $("boxesBtn").setAttribute("aria-pressed", S.boxes); };

  /* ---------- controls ---------- */
  document.querySelectorAll(".seg button").forEach((b) => b.onclick = () => {
    S.detail = b.dataset.detail;
    document.querySelectorAll(".seg button").forEach((x) => { const on = x === b; x.classList.toggle("on", on); x.setAttribute("aria-checked", on); });
  });
  $("speed").oninput = () => { const v = (+$("speed").value).toFixed(1); $("speedOut").textContent = `${v}×`; $("audio").playbackRate = +v; };

  /* ---------- audio player ---------- */
  const audio = $("audio");
  function buildWave(seedText) {
    const w = $("wave"); w.innerHTML = ""; let s = [...seedText].reduce((a, c) => a + c.charCodeAt(0), 7);
    for (let i = 0; i < 56; i++) { s = (s * 9301 + 49297) % 233280; const h = 18 + (s / 233280) * 82; const b = document.createElement("i"); b.style.height = `${h}%`; w.appendChild(b); }
  }
  function setAudio(b64, mime, text) {
    const url = URL.createObjectURL(b64Blob(b64, mime));
    audio.src = url; audio.playbackRate = +$("speed").value;
    $("dlBtn").href = url; $("dlBtn").download = `description.${mime.includes("wav") ? "wav" : "mp3"}`;
    $("player").dataset.disabled = "false"; buildWave(text);
    audio.play().catch(() => toast("Tap play to hear the description."));
    return url;
  }
  audio.ontimeupdate = () => {
    const p = audio.duration ? audio.currentTime / audio.duration : 0, bars = $("wave").children;
    for (let i = 0; i < bars.length; i++) bars[i].classList.toggle("on", i / bars.length < p);
    $("time").textContent = `${fmt(audio.currentTime)} / ${fmt(audio.duration || 0)}`;
  };
  audio.onplay = () => { $("player").classList.add("playing"); $("playBtn").setAttribute("aria-label", "Pause narration"); };
  audio.onpause = audio.onended = () => { $("player").classList.remove("playing"); $("playBtn").setAttribute("aria-label", "Play narration"); };
  $("playBtn").onclick = () => (audio.paused ? audio.play() : audio.pause());
  $("wave").onclick = (e) => { if (!audio.duration) return; const r = $("wave").getBoundingClientRect(); audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration; };
  $("replayBtn").onclick = () => { if (audio.src) { audio.currentTime = 0; audio.play(); } };
  $("copyBtn").onclick = async () => { if (!S.last) return; try { await navigator.clipboard.writeText(S.last.narration); toast("Copied."); } catch { toast("Couldn't copy."); } };

  /* ---------- results ---------- */
  function render(d) {
    S.last = d;
    const cap = $("caption"); cap.classList.remove("placeholder");
    cap.textContent = S.view === "read" ? (d.text ? `“${d.text}”` : "No readable text found.") : d.caption;
    $("timing").textContent = `${(d.timings_ms.total / 1000).toFixed(1)} s`;
    const ob = $("objects"); ob.innerHTML = "";
    if (d.objects.length) d.objects.forEach((o) => { const c = document.createElement("span"); c.className = "chip"; c.textContent = `${o.count} ${o.count > 1 ? plural(o.label) : o.label}`; ob.appendChild(c); });
    else ob.innerHTML = '<span class="muted">Nothing specific</span>';
    $("ocrText").innerHTML = ""; if (d.text) $("ocrText").textContent = d.text; else $("ocrText").innerHTML = '<span class="muted">No text</span>';
    drawBoxes(d.detections, d.ocr);
  }
  const plural = (w) => (d => d[w] || (/(s|x|ch|sh)$/.test(w) ? w + "es" : /[^aeiou]y$/.test(w) ? w.slice(0, -1) + "ies" : w + "s"))({ person: "people", bus: "buses", knife: "knives", mouse: "mice" });

  async function analyze(blob = S.blob, mode = S.view === "read" ? "read" : "describe") {
    if (!blob) return toast("Add an image first.");
    const fd = new FormData(), s = settings();
    fd.append("image", blob, "image.jpg"); fd.append("mode", mode); fd.append("detail", s.detail);
    fd.append("lang", s.lang); fd.append("accent", s.accent);
    if (mode !== "live") { $("goBtn").disabled = true; busy(true, mode === "read" ? "Reading the text…" : "Looking at your image…"); }
    try {
      const r = await fetch(`${API}/api/analyze`, { method: "POST", body: fd });
      const d = await r.json(); if (!r.ok) throw new Error(d.detail || "Something went wrong.");
      return d;
    } catch (e) { toast(e.message === "Failed to fetch" ? "Can't reach the server." : e.message); return null; }
    finally { if (mode !== "live") { busy(false); $("goBtn").disabled = false; } }
  }
  async function runOnce() {
    const d = await analyze(); if (!d) return;
    render(d); setAudio(d.audio_base64, d.audio_mime, d.narration); addHistory(d);
  }

  /* ---------- live mode ---------- */
  async function liveTick() {
    if (S.liveBusy || !S.stream || !audio.paused) return;
    S.liveBusy = true;
    try {
      const blob = await grabFrame();
      const d = await analyze(blob, "live"); if (!d) return;
      S.imgSize = [$("video").videoWidth, $("video").videoHeight];
      if (similar(d.caption, S.lastLiveCaption) < 0.6) { S.lastLiveCaption = d.caption; render(d); setAudio(d.audio_base64, d.audio_mime, d.narration); }
      else drawBoxes(d.detections, []);
    } finally { S.liveBusy = false; }
  }
  function startLive() {
    if (!S.stream) return startCamera().then(() => S.stream && startLive());
    $("liveBadge").hidden = false; $("liveSec").textContent = LIVE_EVERY_MS / 1000; $("goBtn").textContent = "Stop live narration";
    S.lastLiveCaption = ""; liveTick(); S.liveTimer = setInterval(liveTick, LIVE_EVERY_MS);
  }
  function stopLive() { clearInterval(S.liveTimer); S.liveTimer = null; $("liveBadge").hidden = true; stopCamera(); }

  $("goBtn").onclick = () => {
    if (S.view === "live") { if (S.liveTimer) { clearInterval(S.liveTimer); S.liveTimer = null; $("liveBadge").hidden = true; $("goBtn").textContent = VIEW_TEXT.live[0]; } else startLive(); return; }
    runOnce();
  };

  /* ---------- questions ---------- */
  $("askForm").onsubmit = async (e) => {
    e.preventDefault();
    const q = $("ask").value.trim(); $("askErr").hidden = true;
    if (!q) { $("askErr").textContent = "Type a question first."; $("askErr").hidden = false; return; }
    let blob = S.blob; if (S.stream) blob = await grabFrame();
    if (!blob) { $("askErr").textContent = "Add an image first."; $("askErr").hidden = false; return; }
    const fd = new FormData(), s = settings();
    fd.append("image", blob, "image.jpg"); fd.append("question", q); fd.append("lang", s.lang); fd.append("accent", s.accent);
    const item = document.createElement("div"); item.className = "qa";
    item.innerHTML = '<p class="q"></p><p class="a"><span class="spinner" style="width:16px;height:16px;display:inline-block"></span></p>';
    item.querySelector(".q").textContent = `You asked: ${q}`; $("qaList").prepend(item); $("ask").value = "";
    try {
      const r = await fetch(`${API}/api/ask`, { method: "POST", body: fd }); const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "Couldn't answer that.");
      item.querySelector(".a").textContent = d.answer; setAudio(d.audio_base64, d.audio_mime, d.answer);
    } catch (err) { item.querySelector(".a").textContent = err.message; }
  };
  $("ask").oninput = () => ($("askErr").hidden = true);

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) $("micBtn").hidden = true;
  else $("micBtn").onclick = () => {
    const rec = new SR(); rec.lang = $("lang").value === "en" ? "en-IN" : $("lang").value; rec.interimResults = false;
    $("micBtn").classList.add("rec"); rec.start();
    rec.onresult = (e) => { $("ask").value = e.results[0][0].transcript; $("askForm").requestSubmit(); };
    rec.onend = () => $("micBtn").classList.remove("rec");
    rec.onerror = () => toast("Didn't catch that. Try again.");
  };

  /* ---------- history ---------- */
  try { S.history = JSON.parse(localStorage.getItem("v2v_history") || "[]"); } catch { S.history = []; }
  const saveHistory = () => { try { localStorage.setItem("v2v_history", JSON.stringify(S.history.slice(0, 30))); } catch {} };
  function thumb() {
    const img = $("preview"); if (!img.naturalWidth) return "";
    const c = document.createElement("canvas"), k = 160 / Math.max(img.naturalWidth, img.naturalHeight);
    c.width = img.naturalWidth * k; c.height = img.naturalHeight * k; c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.7);
  }
  function addHistory(d) { S.history.unshift({ t: Date.now(), thumb: thumb(), caption: d.caption, narration: d.narration, lang: d.lang }); saveHistory(); renderRecent(); }
  function histItem(h) {
    const b = document.createElement("button"); b.className = "hist-item";
    const im = document.createElement("img"); im.alt = ""; if (h.thumb) im.src = h.thumb;
    const sp = document.createElement("span"); sp.textContent = h.caption;
    b.append(im, sp); b.title = new Date(h.t).toLocaleString();
    b.onclick = async () => {
      const fd = new FormData(); fd.append("text", h.narration); fd.append("lang", "en"); fd.append("accent", $("accent").value);
      try { const r = await fetch(`${API}/api/tts`, { method: "POST", body: fd }); const d = await r.json(); setAudio(d.audio_base64, d.audio_mime, h.narration); toast("Playing saved description."); } catch { toast("Can't reach the server."); }
    };
    return b;
  }
  function renderRecent() { const r = $("recent"); r.innerHTML = ""; S.history.slice(0, 4).forEach((h) => r.appendChild(histItem(h))); }
  function renderHistory() { const g = $("histGrid"); g.innerHTML = ""; S.history.forEach((h) => g.appendChild(histItem(h))); $("histEmpty").hidden = S.history.length > 0; }
  $("clearHist").onclick = () => { S.history = []; saveHistory(); renderHistory(); renderRecent(); };
  renderRecent();

  /* keyboard shortcut: Space plays/pauses when not typing */
  document.addEventListener("keydown", (e) => { if (e.code === "Space" && !/INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement.tagName) && document.activeElement !== $("stage") && audio.src) { e.preventDefault(); audio.paused ? audio.play() : audio.pause(); } });
})();
