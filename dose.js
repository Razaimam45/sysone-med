// SysOne-Med · Dose or Hold — medication timing.
// A glucose trace scrolls and a dosing beat arrives at the decision line. The
// level is ABOVE TARGET (-> GIVE) or IN RANGE (-> HOLD). All five models face
// the same beat, but each decides at its MEASURED latency and with its
// MEASURED accuracy. A correct, in-time call keeps the patient stable; a wrong
// or too-slow call — an over-dose or a missed dose — drains that model's
// stability bar. When a bar hits zero the model flatlines, so both accuracy
// and speed decide who survives.
const $ = id => document.getElementById(id);
const TIME_SCALE = 3;   // visual ms per real ms
const MAX_BEATS = 60;   // shift length

let variants = [];
let windowMs = 400;
let mult = 0.5;
let raf = null, playing = false, lastFrame = 0;
let beatIndex = 0;
let beat = null;        // {label, isGive, eff, travel, start, resolved:{}}
let models = {};        // id -> {v, stability, alive, correct, total, survived}
let gPhase = 0, gBuf = [];

const clamp01 = x => Math.max(0.02, Math.min(0.999, x));

function nextBeat() {
    const isGive = Math.random() < 0.5;
    const val = isGive ? Math.round(150 + Math.random() * 180) : Math.round(80 + Math.random() * 55);
    return {
        label: 'GLU ' + val + ' mg/dL',
        isGive,
        resolved: {}
    };
}

// The window tightens as more beats arrive (decision fatigue).
function effWindow() {
    return windowMs * Math.max(0.55, 1 - 0.015 * beatIndex);
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
    spawnBeat();
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

// ---------- beat lifecycle ----------
function spawnBeat() {
    beatIndex++;
    beat = nextBeat();
    beat.eff = effWindow();
    beat.travel = beat.eff * TIME_SCALE / mult;
    beat.start = performance.now();
    const m = $('marker');
    m.className = 'beat-marker ' + (beat.isGive ? 'give' : 'hold');
    m.style.left = '100%';
    m.style.opacity = 1;
    $('markerLabel').textContent = beat.label;
    $('traceTag').textContent = beat.isGive ? 'ABOVE TARGET → GIVE' : 'IN RANGE → HOLD';
    $('traceNum').textContent = 'BEAT ' + String(beatIndex).padStart(4, '0');
    for (const id in models) {
        if (!models[id].alive) continue;
        const el = $('model-' + id);
        const vd = el.querySelector('.m-verdict');
        vd.textContent = 'DECIDING…';
        vd.className = 'm-verdict';
    }
}

function stepBeat(now) {
    const t = Math.min(1, (now - beat.start) / beat.travel);
    $('marker').style.left = (100 - t * 22) + '%';
    for (const id in models) {
        const M = models[id];
        if (!M.alive || beat.resolved[id]) continue;
        const frac = M.v.mean_latency_ms / beat.eff;
        if (frac >= 1) {
            if (t >= 1) resolveModel(id, true);
        } else if (t >= frac) {
            resolveModel(id, false);
        }
    }
    if (t >= 1) {
        for (const id in models) if (models[id].alive && !beat.resolved[id]) resolveModel(id, true);
        $('marker').style.opacity = 0;
        beat = null;
        setTimeout(() => {
            if (!playing) return;
            if (allDead() || beatIndex >= MAX_BEATS) { endGame(); return; }
            spawnBeat();
        }, 650 / mult);
    }
}

function resolveModel(id, late) {
    const M = models[id];
    beat.resolved[id] = true;
    M.total++;
    const wouldBeOk = Math.random() * 100 < M.v.accuracy;
    const action = wouldBeOk ? (beat.isGive ? 'GIVE' : 'HOLD') : (beat.isGive ? 'HOLD' : 'GIVE');
    const ok = late ? false : wouldBeOk;
    if (ok) { M.correct++; M.stability = Math.min(100, M.stability + 4); }
    else { M.stability -= 14; }
    const conf = clamp01(M.v.conf_mean + (Math.random() - 0.5) * 0.15);
    const lat = M.v.mean_latency_ms * (0.9 + Math.random() * 0.2);
    const el = $('model-' + id);
    const vd = el.querySelector('.m-verdict');
    if (M.stability <= 0) {
        M.stability = 0; M.alive = false; M.survived = beatIndex;
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
    addStream(M.v, beat.label, action, conf, lat, ok, late ? 'TOO SLOW' : (ok ? 'CORRECT' : 'WRONG'));
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
      <div class="r-sub">${M.alive ? 'all ' + beatIndex + ' beats · ' + M.correct + '/' + M.total + ' correct'
        : M.correct + '/' + M.total + ' correct before flatline'}</div>
    </div>`).join('');
    $('results').hidden = false;
    $('results').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ---------- stream + stats ----------
function addStream(v, beatLabel, action, conf, lat, ok, outcome) {
    const el = document.createElement('div');
    el.className = 'decision' + (ok ? '' : ' bad');
    el.innerHTML = `<span class="d-left"><span class="d-sw" style="background:${v.color}"></span>
      <span class="d-name">${v.name}</span><span class="d-act">${beatLabel} → ${action}</span></span>
    <span class="d-right"><span class="d-conf">${conf.toFixed(2)}</span>
      <span class="d-lat">${Math.round(lat)}ms</span><span class="d-out">${outcome}</span></span>`;
    const stream = $('stream');
    stream.prepend(el);
    while (stream.children.length > 60) stream.removeChild(stream.lastChild);
}

function updateStats() {
    const alive = Object.values(models).filter(M => M.alive);
    $('stBeats').textContent = beatIndex;
    $('stAlive').textContent = alive.length;
    let leader = '—';
    if (alive.length) {
        leader = alive.reduce((a, b) => a.stability >= b.stability ? a : b).v.name;
    }
    $('stLeader').textContent = leader;
    $('stWindow').textContent = windowMs + ' ms';
}

// ---------- glucose canvas ----------
function sizeCanvas() {
    const cv = $('glucose');
    const dpr = window.devicePixelRatio || 1;
    const r = cv.getBoundingClientRect();
    cv.width = Math.max(10, Math.floor(r.width * dpr));
    cv.height = Math.max(10, Math.floor(r.height * dpr));
    gBuf = new Array(cv.width).fill(0.5);
}

// Smooth wandering glucose in [0,1]; 0.5 ≈ mid target band.
function glucoseSample(p) {
    return 0.5
        + 0.28 * Math.sin(p * 0.9)
        + 0.16 * Math.sin(p * 2.3 + 1.1)
        + 0.08 * Math.sin(p * 5.1 + 0.4)
        + (Math.random() - 0.5) * 0.02;
}

function advanceGlucose(dt) {
    const pxPerMs = 0.2 * mult;
    const n = Math.floor(pxPerMs * dt);
    if (n <= 0) return;
    const phasePerPx = 1 / 220;
    for (let i = 0; i < n; i++) {
        gPhase = (gPhase + phasePerPx) % 1000;
        gBuf.push(glucoseSample(gPhase));
    }
    const cv = $('glucose');
    while (gBuf.length > cv.width) gBuf.shift();
}

function drawGlucose() {
    const cv = $('glucose');
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#0e1a24';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x < W; x += 28) { ctx.moveTo(x, 0); ctx.lineTo(x, H); }
    for (let y = 0; y < H; y += 28) { ctx.moveTo(0, y); ctx.lineTo(W, y); }
    ctx.stroke();
    const mid = H * 0.5, amp = H * 0.42;
    ctx.strokeStyle = '#ffb454';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#ffb454';
    ctx.shadowBlur = 6;
    ctx.beginPath();
    const n = gBuf.length;
    for (let i = 0; i < n; i++) {
        const x = W - (n - i);
        const y = mid - gBuf[i] * amp;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
}

// ---------- main loop ----------
function loop(now) {
    const dt = Math.min(50, now - lastFrame);
    lastFrame = now;
    advanceGlucose(dt);
    drawGlucose();
    if (playing && beat) stepBeat(now);
    raf = requestAnimationFrame(loop);
}

// ---------- controls ----------
$('play').onclick = () => {
    if (playing) { playing = false; $('play').textContent = '▶ Play'; return; }
    playing = true; $('play').textContent = 'Ⅱ Pause';
    if (!beat) spawnBeat();
};

$('reset').onclick = () => {
    playing = false; $('play').textContent = '▶ Play';
    beatIndex = 0; beat = null;
    $('stream').innerHTML = '';
    $('results').hidden = true;
    $('marker').style.opacity = 0;
    setupModels();
    buildModels();
    updateStats();
    spawnBeat();
};

$('window').oninput = e => {
    windowMs = Number(e.target.value);
    $('windowVal').textContent = windowMs + ' ms';
    $('stWindow').textContent = windowMs + ' ms';
};

$('mult').onchange = e => { mult = Number(e.target.value); };

load();
