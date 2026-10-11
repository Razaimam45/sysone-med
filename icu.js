// SysOne-Med · Alarm Storm — ICU throughput race.
// Ten ICU patients alarm at once (an alarm storm). Five triage teams race to
// clear the alarms, each one triaging a patient at its MEASURED mean latency.
// In a fixed shift, a team clears (shift / latency) patients — so the fast
// paths clear the whole storm and the slow ones can barely keep up. This is a
// THROUGHPUT race: in a storm, the model that can't triage fast is useless,
// no matter how accurate it is. The decision stream logs every triage.
const $ = id => document.getElementById(id);
const TIME_SCALE = 3;   // visual ms per real ms
const N_BEDS = 10;

let variants = [];      // sorted fastest -> slowest
let shiftMs = 800;
let mult = 0.5;
let storms = 0, raf = null, last = 0, playing = false;
let shiftStart = 0, shiftDur = 0;
let beds = [];

const clamp01 = x => Math.max(0.02, Math.min(0.999, x));
const pick = arr => arr[Math.floor(Math.random() * arr.length)];

const ALARMS = [
    { ico: '🫁', label: 'SpO₂ low' },
    { ico: '❤️', label: 'Tachycardia' },
    { ico: '🩸', label: 'Hypotension' },
    { ico: '🌡️', label: 'Fever' },
    { ico: '⚡', label: 'Arrhythmia' },
    { ico: '💧', label: 'Fluid overload' },
    { ico: '🧠', label: 'GCS drop' },
    { ico: '🫀', label: 'Low output' },
    { ico: '🩻', label: 'Vent alarm' },
    { ico: '📉', label: 'MAP low' },
];

// ---------- pilot / variant setup ----------
async function load() {
    let cfg;
    try {
        const r = await fetch('data/triage.json');
        if (!r.ok) throw new Error('HTTP ' + r.status);
        cfg = await r.json();
    } catch (e) {
        $('teams').innerHTML = '<p class="muted">Could not load variants: ' + e.message + '</p>';
        return;
    }
    variants = cfg.variants.slice().sort((a, b) => a.mean_latency_ms - b.mean_latency_ms);
    buildTeams();
    newStorm();
}

function buildTeams() {
    $('teams').innerHTML = variants.map(v => `
    <div class="team" id="team-${v.id}" style="--c:${v.color}">
      <div class="t-head"><span class="t-sw"></span><span class="t-name">${v.name}</span>
        <span class="t-lat">${Math.round(v.mean_latency_ms)} ms</span></div>
      <div class="t-count">0 <small>/ 10</small></div>
      <div class="t-lbl">alarms cleared</div>
      <div class="t-bar"><i></i></div>
      <div class="t-foot"><span>conf <b class="t-conf">${v.conf_mean != null ? v.conf_mean.toFixed(2) : '—'}</b></span><span class="t-out">—</span></div>
    </div>`).join('');
}

// ---------- storm lifecycle ----------
function newStorm() {
    storms++;
    $('stormNum').textContent = 'STORM ' + String(storms).padStart(4, '0');
    beds = [];
    for (let i = 0; i < N_BEDS; i++) {
        const a = pick(ALARMS);
        beds.push({ id: 'B' + (i + 1), ico: a.ico, label: a.label });
    }
    renderBeds();
    for (const v of variants) {
        v.count = Math.min(N_BEDS, Math.floor(shiftMs / v.mean_latency_ms));
        v.interval = v.mean_latency_ms * TIME_SCALE / mult;
        v.cleared = 0;
    }
    for (const v of variants) {
        const el = $('team-' + v.id);
        el.classList.remove('done');
        el.querySelector('.t-count').innerHTML = '0 <small>/ 10</small>';
        el.querySelector('.t-bar i').style.width = '0%';
        el.querySelector('.t-out').textContent = '—';
    }
    shiftDur = shiftMs * TIME_SCALE / mult;
    shiftStart = performance.now();
    if (playing) { last = shiftStart; raf = requestAnimationFrame(tick); }
}

function renderBeds() {
    const c = variants[0] ? variants[0].cleared : 0;
    $('beds').innerHTML = beds.map((b, i) => {
        let state = 'alarming';
        if (i < c) state = 'stable';
        else if (i === c && c < N_BEDS) state = 'triaging';
        const lbl = state === 'stable' ? 'STABLE' : state === 'triaging' ? 'TRIAGING' : 'ALARMING';
        return `<div class="bed ${state}" id="bed-${b.id}">
          <div class="b-top"><span class="b-id">${b.id}</span><span class="b-ico">${b.ico}</span></div>
          <div class="b-state">${lbl}</div>
          <div class="b-bar"><i></i></div>
        </div>`;
    }).join('');
}

function tick(now) {
    if (!playing) return;
    const elapsed = now - shiftStart;
    const t = Math.min(1, elapsed / shiftDur);
    for (const v of variants) {
        if (v.cleared >= v.count) continue;
        const target = Math.min(v.count, Math.floor(elapsed / v.interval));
        while (v.cleared < target) {
            v.cleared++;
            clearPatient(v);
        }
    }
    if (t >= 1) { endShift(); return; }
    raf = requestAnimationFrame(tick);
}

function clearPatient(v) {
    const conf = v.conf_mean != null ? clamp01(v.conf_mean + (Math.random() - 0.5) * 0.15) : null;
    const lat = v.mean_latency_ms * (0.9 + Math.random() * 0.2);
    const a = pick(ALARMS);
    const el = $('team-' + v.id);
    el.querySelector('.t-count').innerHTML = v.cleared + ' <small>/ 10</small>';
    el.querySelector('.t-bar i').style.width = (v.cleared / N_BEDS * 100) + '%';
    if (v === variants[0]) renderBeds();
    addStream(v, a.label, conf, lat, true, 'CLEARED');
}

function endShift() {
    for (const v of variants) {
        const el = $('team-' + v.id);
        el.classList.add('done');
        const full = v.count >= N_BEDS;
        el.querySelector('.t-out').textContent = full ? 'FULL' : v.count + '/10';
        addStream(v, v.count + '/10 cleared', null, v.mean_latency_ms, full, v.count + '/10');
    }
    updateStats();
    setTimeout(() => { if (playing) newStorm(); }, 1000 / mult);
}

function addStream(v, action, conf, lat, saved, outLabel) {
    const el = document.createElement('div');
    el.className = 'decision ' + (saved ? 'saved' : 'tooslow');
    el.innerHTML = `<span class="d-left"><span class="d-sw" style="background:${v.color}"></span>
      <span class="d-name">${v.name}</span><span class="d-act">${action}</span></span>
    <span class="d-right">${conf != null ? `<span class="d-conf">${conf.toFixed(2)}</span>` : ''}
      <span class="d-lat">${Math.round(lat)}ms</span><span class="d-out">${outLabel || (saved ? 'CLEARED' : 'BEHIND')}</span></span>`;
    const stream = $('stream');
    stream.prepend(el);
    while (stream.children.length > 60) stream.removeChild(stream.lastChild);
}

function updateStats() {
    $('stShifts').textContent = storms;
    const best = variants[0] ? variants[0].count : 0;
    const worst = variants.length ? variants[variants.length - 1].count : 0;
    $('stBest').textContent = best + '/10';
    $('stWorst').textContent = worst + '/10';
    $('stSpread').textContent = variants.length ?
        Math.round(variants[variants.length - 1].mean_latency_ms / variants[0].mean_latency_ms) + '×' : '—';
}

// ---------- controls ----------
function togglePlay() {
    if (playing) { playing = false; cancelAnimationFrame(raf); $('play').textContent = '▶ Play'; return; }
    playing = true; $('play').textContent = 'Ⅱ Pause';
    last = performance.now();
    raf = requestAnimationFrame(tick);
}

$('play').onclick = togglePlay;
$('reset').onclick = () => { playing = false; cancelAnimationFrame(raf); $('play').textContent = '▶ Play'; storms = 0; $('stream').innerHTML = ''; newStorm(); };
$('shift').oninput = e => {
    shiftMs = Number(e.target.value);
    $('shiftVal').textContent = shiftMs + ' ms';
    if (!playing) {
        for (const v of variants) {
            v.count = Math.min(N_BEDS, Math.floor(shiftMs / v.mean_latency_ms));
            v.cleared = 0;
        }
        shiftDur = shiftMs * TIME_SCALE / mult;
        renderBeds();
        for (const v of variants) {
            const el = $('team-' + v.id);
            el.querySelector('.t-count').innerHTML = '0 <small>/ 10</small>';
            el.querySelector('.t-bar i').style.width = '0%';
        }
    }
};
$('mult').onchange = e => { mult = Number(e.target.value); };
document.addEventListener('keydown', e => { if (e.code === 'Space') { e.preventDefault(); togglePlay(); } });

load();
