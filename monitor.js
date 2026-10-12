// SysOne-Med · The Monitor — alarm interpretation.
// A telemetry strip scrolls and alarms slide toward the decision line. Each
// alarm is TRUE (-> ACT) or FALSE (-> HOLD). All five models face the same
// alarm, but each decides at its MEASURED latency and with its MEASURED
// accuracy. A correct, in-time call keeps the patient stable; a wrong or
// too-slow call drains that model's stability bar. When a bar hits zero the
// model flatlines — so both accuracy and speed decide who survives.
const $ = id => document.getElementById(id);
const TIME_SCALE = 3;   // visual ms per real ms
const MAX_ALARMS = 60;  // shift length

let variants = [];
let windowMs = 400;
let mult = 0.5;
let raf = null, playing = false, lastFrame = 0;
let alarmIndex = 0;
let alarm = null;        // {label, isTrue, eff, travel, start, resolved:{}}
let models = {};         // id -> {v, stability, alive, correct, total, survived}
let ecgPhase = 0, ecgBuf = [];

const clamp01 = x => Math.max(0.02, Math.min(0.999, x));

const TRUE_ALARMS = [
    'VTACH · 182 bpm', 'ASYSTOLE', 'SPo₂ 84% · LOW', 'APNEA · 12 s',
    'ST ELEVATION · II,III,aVF', 'SVT · 210 bpm', 'BRADY · 31 bpm', 'ETCO₂ 48 · HIGH'
];
const FALSE_ALARMS = [
    'ARTIFACT · LOOSE LEAD II', 'COUGH ARTIFACT', 'MOTION ARTIFACT · V1',
    'SENSOR DRIFT · SPo₂', 'FALSE HIGH HR · 168', 'LEAD OFF · aVF', 'CABLE TUG · V5'
];

function nextAlarm() {
    const isTrue = Math.random() < 0.5;
    const pool = isTrue ? TRUE_ALARMS : FALSE_ALARMS;
    return { label: pool[Math.floor(Math.random() * pool.length)], isTrue, resolved: {} };
}

// The window tightens as more alarms arrive (alarm fatigue).
function effWindow() {
    return windowMs * Math.max(0.55, 1 - 0.015 * alarmIndex);
}

// ---------- setup ----------
async function load() {
    let cfg;
    try {
        const r = await fetch('data/triage.json');
        if (!r.ok) throw new Error('HTTP ' + r.status);
        cfg = await r.json();
    } catch (e) {
        $('models').innerHTML = '<p class="muted">Could not load variants: ' + e.message + '</p>';
        return;
    }
    variants = cfg.variants.slice().sort((a, b) => a.mean_latency_ms - b.mean_latency_ms);
    buildModels();
    setupModels();
    sizeCanvas();
    window.addEventListener('resize', sizeCanvas);
    lastFrame = performance.now();
    raf = requestAnimationFrame(loop);
    spawnAlarm();
}

function setupModels() {
    models = {};
    for (const v of variants) {
        models[v.id] = { v, stability: 100, alive: true, correct: 0, total: 0, survived: 0 };
    }
}

function buildModels() {
    $('models').innerHTML = variants.map(v => `
    <div class="model" id="model-${v.id}" style="--c:${v.color}">
      <div class="m-head"><span class="m-sw"></span><span class="m-name">${v.name}</span>
        <span class="m-lat">${Math.round(v.mean_latency_ms)} ms</span></div>
      <div class="m-verdict">STANDING BY</div>
      <div class="stab-row"><div class="stab-bar"><i></i></div><span class="stab-pct">100%</span></div>
      <div class="m-foot"><span>conf <b>${v.conf_mean != null ? v.conf_mean.toFixed(2) : '—'}</b></span>
        <span><b class="m-correct">0/0</b> correct</span></div>
    </div>`).join('');
}

// ---------- alarm lifecycle ----------
function spawnAlarm() {
    alarmIndex++;
    alarm = nextAlarm();
    alarm.eff = effWindow();
    alarm.travel = alarm.eff * TIME_SCALE / mult;
    alarm.start = performance.now();
    const m = $('marker');
    m.className = 'event-marker ' + (alarm.isTrue ? 'true' : 'hold');
    m.style.left = '100%';
    m.style.opacity = 1;
    $('markerLabel').textContent = alarm.label;
    $('teleTag').textContent = alarm.isTrue ? 'TRUE ALARM → ACT' : 'FALSE ALARM → HOLD';
    $('teleNum').textContent = 'ALARM ' + String(alarmIndex).padStart(4, '0');
    for (const id in models) {
        if (!models[id].alive) continue;
        const el = $('model-' + id);
        const vd = el.querySelector('.m-verdict');
        vd.textContent = 'DECIDING…';
        vd.className = 'm-verdict';
    }
}

function stepAlarm(now) {
    const t = Math.min(1, (now - alarm.start) / alarm.travel);
    $('marker').style.left = (100 - t * 22) + '%';
    for (const id in models) {
        const M = models[id];
        if (!M.alive || alarm.resolved[id]) continue;
        const frac = M.v.mean_latency_ms / alarm.eff;
        if (frac >= 1) {
            if (t >= 1) resolveModel(id, true);
        } else if (t >= frac) {
            resolveModel(id, false);
        }
    }
    if (t >= 1) {
        for (const id in models) if (models[id].alive && !alarm.resolved[id]) resolveModel(id, true);
        $('marker').style.opacity = 0;
        alarm = null;
        setTimeout(() => {
            if (!playing) return;
            if (allDead() || alarmIndex >= MAX_ALARMS) { endGame(); return; }
            spawnAlarm();
        }, 650 / mult);
    }
}

function resolveModel(id, late) {
    const M = models[id];
    alarm.resolved[id] = true;
    M.total++;
    const wouldBeOk = Math.random() * 100 < M.v.accuracy;
    const action = wouldBeOk ? (alarm.isTrue ? 'ACT' : 'HOLD') : (alarm.isTrue ? 'HOLD' : 'ACT');
    const ok = late ? false : wouldBeOk;
    if (ok) { M.correct++; M.stability = Math.min(100, M.stability + 4); }
    else { M.stability -= 14; }
    const conf = clamp01(M.v.conf_mean + (Math.random() - 0.5) * 0.15);
    const lat = M.v.mean_latency_ms * (0.9 + Math.random() * 0.2);
    const el = $('model-' + id);
    const vd = el.querySelector('.m-verdict');
    if (M.stability <= 0) {
        M.stability = 0; M.alive = false; M.survived = alarmIndex;
        el.classList.add('dead');
        vd.textContent = '✕ FLATLINED';
        vd.className = 'm-verdict dead';
    } else {
        vd.textContent = (ok ? '✓ ' : '✕ ') + action + (late ? ' · LATE' : '');
        vd.className = 'm-verdict ' + (ok ? 'good' : 'bad');
    }
    el.querySelector('.stab-bar i').style.width = Math.max(0, M.stability) + '%';
    el.querySelector('.stab-bar i').classList.toggle('low', M.stability < 35);
    el.querySelector('.stab-pct').textContent = Math.round(Math.max(0, M.stability)) + '%';
    el.querySelector('.m-correct').textContent = M.correct + '/' + M.total;
    addStream(M.v, alarm.label, action, conf, lat, ok, late ? 'TOO SLOW' : (ok ? 'CORRECT' : 'WRONG'));
    updateStats();
}

function allDead() {
    return Object.values(models).every(M => !M.alive);
}

function endGame() {
    playing = false;
    $('play').textContent = '▶ Play';
    const ranked = Object.values(models).sort((a, b) => (b.alive - a.alive) || (b.stability - a.stability));
    $('resultsGrid').innerHTML = ranked.map((M, i) => `
    <div class="res-card ${i === 0 && M.alive ? 'winner' : ''}">
      <div class="r-name" style="color:${M.v.color}">${M.v.name}</div>
      <div class="r-big">${M.alive ? 'ALIVE' : 'FLAT @ ' + M.survived}</div>
      <div class="r-sub">${M.alive ? 'all ' + alarmIndex + ' alarms · ' + M.correct + '/' + M.total + ' correct'
        : M.correct + '/' + M.total + ' correct before flatline'}</div>
    </div>`).join('');
    $('results').hidden = false;
    $('results').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ---------- stream + stats ----------
function addStream(v, alarmLabel, action, conf, lat, ok, outcome) {
    const el = document.createElement('div');
    el.className = 'decision' + (ok ? '' : ' bad');
    el.innerHTML = `<span class="d-left"><span class="d-sw" style="background:${v.color}"></span>
      <span class="d-name">${v.name}</span><span class="d-act">${alarmLabel} → ${action}</span></span>
    <span class="d-right"><span class="d-conf">${conf.toFixed(2)}</span>
      <span class="d-lat">${Math.round(lat)}ms</span><span class="d-out">${outcome}</span></span>`;
    const stream = $('stream');
    stream.prepend(el);
    while (stream.children.length > 60) stream.removeChild(stream.lastChild);
}

function updateStats() {
    const alive = Object.values(models).filter(M => M.alive);
    $('stEvents').textContent = alarmIndex;
    $('stAlive').textContent = alive.length;
    let leader = '—';
    if (alive.length) {
        leader = alive.reduce((a, b) => a.stability >= b.stability ? a : b).v.name;
    }
    $('stLeader').textContent = leader;
    $('stWindow').textContent = windowMs + ' ms';
}

// ---------- ECG canvas ----------
function sizeCanvas() {
    const cv = $('ecg');
    const dpr = window.devicePixelRatio || 1;
    const r = cv.getBoundingClientRect();
    cv.width = Math.max(10, Math.floor(r.width * dpr));
    cv.height = Math.max(10, Math.floor(r.height * dpr));
    ecgBuf = new Array(cv.width).fill(0);
}

function ecgSample(p) {
    let y = 0;
    if (p > 0.10 && p < 0.22) y += 0.10 * Math.sin((p - 0.10) / 0.12 * Math.PI);
    if (p > 0.28 && p < 0.32) y -= 0.12 * Math.sin((p - 0.28) / 0.04 * Math.PI);
    if (p > 0.32 && p < 0.38) y += 0.95 * Math.sin((p - 0.32) / 0.06 * Math.PI);
    if (p > 0.38 && p < 0.42) y -= 0.28 * Math.sin((p - 0.38) / 0.04 * Math.PI);
    if (p > 0.55 && p < 0.75) y += 0.22 * Math.sin((p - 0.55) / 0.20 * Math.PI);
    return y;
}

function advanceEcg(dt) {
    const pxPerMs = 0.2 * mult;
    const n = Math.floor(pxPerMs * dt);
    if (n <= 0) return;
    const phasePerPx = 1 / 140;
    for (let i = 0; i < n; i++) {
        ecgPhase = (ecgPhase + phasePerPx) % 1;
        ecgBuf.push(ecgSample(ecgPhase) + (Math.random() - 0.5) * 0.02);
    }
    const cv = $('ecg');
    while (ecgBuf.length > cv.width) ecgBuf.shift();
}

function drawEcg() {
    const cv = $('ecg');
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#0e1a24';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x < W; x += 28) { ctx.moveTo(x, 0); ctx.lineTo(x, H); }
    for (let y = 0; y < H; y += 28) { ctx.moveTo(0, y); ctx.lineTo(W, y); }
    ctx.stroke();
    const mid = H * 0.55, amp = H * 0.32;
    ctx.strokeStyle = '#37e0a8';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#37e0a8';
    ctx.shadowBlur = 6;
    ctx.beginPath();
    const n = ecgBuf.length;
    for (let i = 0; i < n; i++) {
        const x = W - (n - i);
        const y = mid - ecgBuf[i] * amp;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
}

// ---------- main loop ----------
function loop(now) {
    const dt = Math.min(50, now - lastFrame);
    lastFrame = now;
    advanceEcg(dt);
    drawEcg();
    if (playing && alarm) stepAlarm(now);
    raf = requestAnimationFrame(loop);
}

// ---------- controls ----------
$('play').onclick = () => {
    if (playing) { playing = false; $('play').textContent = '▶ Play'; return; }
    playing = true; $('play').textContent = 'Ⅱ Pause';
    if (!alarm) spawnAlarm();
};

$('reset').onclick = () => {
    playing = false; $('play').textContent = '▶ Play';
    alarmIndex = 0; alarm = null;
    $('stream').innerHTML = '';
    $('results').hidden = true;
    $('marker').style.opacity = 0;
    setupModels();
    buildModels();
    updateStats();
    spawnAlarm();
};

$('window').oninput = e => {
    windowMs = Number(e.target.value);
    $('windowVal').textContent = windowMs + ' ms';
    $('stWindow').textContent = windowMs + ' ms';
};

$('mult').onchange = e => { mult = Number(e.target.value); };

load();
