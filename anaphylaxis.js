// SysOne-Med · The Closing Door — anaphylaxis airway race.
// A patient goes into anaphylactic shock and the airway closes in real time.
// All five decision paths race to inject epinephrine before the airway shuts.
// Each path takes its MEASURED mean latency: if it lands while the airway is
// still open it saves the patient, if the door shuts first it's a code blue.
// Speed is the whole game — accuracy is not required to win the race.
const $ = id => document.getElementById(id);
const TIME_SCALE = 3;   // visual ms per real ms, so a 600ms closure is watchable

let variants = [];      // sorted fastest -> slowest
let closureMs = 600;
let mult = 0.5;
let patients = 0, raf = null, last = 0, playing = false;
let caseStart = 0, caseDur = 0, resolved = {};

const clamp01 = x => Math.max(0.02, Math.min(0.999, x));
const fmtMs = ms => ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : Math.round(ms) + ' ms';

// ---------- anaphylaxis triggers ----------
const TRIGGERS = [
    'Peanut exposure', 'Hymenoptera sting', 'IV contrast', 'Penicillin',
    'Latex contact', 'Shellfish', 'Aspirin', 'Unknown trigger'
];
const ri = (a, b) => a + Math.floor(Math.random() * (b - a + 1));
const rf = (a, b) => (a + Math.random() * (b - a)).toFixed(1);
const pick = arr => arr[Math.floor(Math.random() * arr.length)];

function nextPatient() {
    return pick(TRIGGERS) + ' · Stridor ' + ri(70, 95) + ' dB · SpO₂ ' + ri(78, 92) +
        '% · HR ' + ri(110, 150) + ' · Urticaria ' + ri(2, 5) + '/5';
}

// ---------- pilot / variant setup ----------
async function load() {
    let cfg;
    try {
        const r = await fetch('data/triage.json');
        if (!r.ok) throw new Error('HTTP ' + r.status);
        cfg = await r.json();
    } catch (e) {
        $('racers').innerHTML = '<p class="muted">Could not load variants: ' + e.message + '</p>';
        return;
    }
    variants = cfg.variants.slice().sort((a, b) => a.mean_latency_ms - b.mean_latency_ms);
    buildRacers();
    newPatient();
}

function buildRacers() {
    $('racers').innerHTML = variants.map(v => `
    <div class="racer" id="racer-${v.id}" style="--c:${v.color}">
      <div class="r-head"><span class="r-sw"></span><span class="r-name">${v.name}</span>
        <span class="r-lat">${Math.round(v.mean_latency_ms)} ms</span></div>
      <div class="r-verdict waiting">STANDING BY</div>
      <div class="r-bar"><i></i></div>
      <div class="r-foot"><span>conf <b class="r-conf">—</b></span><span class="r-out">—</span></div>
    </div>`).join('');
}

// ---------- patient lifecycle ----------
function newPatient() {
    patients++;
    resolved = {};
    $('caseNum').textContent = 'PATIENT ' + String(patients).padStart(4, '0');
    $('caseState').textContent = nextPatient();
    // reset racer cards
    for (const v of variants) {
        const el = $('racer-' + v.id);
        el.classList.remove('saved', 'tooslow');
        el.querySelector('.r-verdict').textContent = 'DECIDING…';
        el.querySelector('.r-verdict').className = 'r-verdict waiting';
        el.querySelector('.r-bar i').style.width = '0%';
        el.querySelector('.r-conf').textContent = '—';
        el.querySelector('.r-out').textContent = '—';
    }
    // reset airway
    $('airway').classList.remove('critical', 'closed');
    renderAirway(0);
    // visual duration of this patient
    caseDur = closureMs * TIME_SCALE / mult;
    caseStart = performance.now();
    if (playing) { last = caseStart; raf = requestAnimationFrame(tick); }
}

function tick(now) {
    if (!playing) return;
    const elapsed = now - caseStart;
    const t = Math.min(1, elapsed / caseDur);
    renderAirway(t);
    // resolve each racer as its (scaled) latency is reached
    for (const v of variants) {
        if (resolved[v.id]) continue;
        const frac = v.mean_latency_ms / closureMs;   // where it injects relative to closure
        if (frac >= 1) {
            if (t >= 1) resolveRacer(v, false, now);
        } else if (t >= frac) {
            resolveRacer(v, true, now);
        }
    }
    if (t >= 1) {
        for (const v of variants) if (!resolved[v.id]) resolveRacer(v, false, now);
        setTimeout(() => { if (playing) newPatient(); }, 900 / mult);
        return;
    }
    raf = requestAnimationFrame(tick);
}

function resolveRacer(v, saved, now) {
    resolved[v.id] = true;
    const conf = v.conf_mean != null ? clamp01(v.conf_mean + (Math.random() - 0.5) * 0.15) : null;
    const lat = v.mean_latency_ms * (0.9 + Math.random() * 0.2);
    const el = $('racer-' + v.id);
    el.classList.add(saved ? 'saved' : 'tooslow');
    const verdict = el.querySelector('.r-verdict');
    verdict.textContent = saved ? '✓ EPINEPHRINE IN' : '✕ CODE BLUE';
    verdict.className = 'r-verdict ' + (saved ? 'saved' : 'tooslow');
    el.querySelector('.r-bar i').style.width = (saved ? Math.min(100, lat / closureMs * 100) : 100) + '%';
    el.querySelector('.r-conf').textContent = conf != null ? conf.toFixed(2) : '—';
    el.querySelector('.r-out').textContent = Math.round(lat) + ' ms';
    addStream(v, conf, lat, saved);
    updateStats();
}

function renderAirway(t) {
    const patency = Math.max(0, Math.round((1 - t) * 100));
    // doors slide in from the sides as t -> 1
    $('doorL').style.transform = 'translateX(' + (-100 * (1 - t)) + '%)';
    $('doorR').style.transform = 'translateX(' + (100 * (1 - t)) + '%)';
    $('patency').textContent = patency + '%';
    const breath = $('breath');
    breath.style.transform = 'scale(' + (0.5 + 0.5 * (1 - t)) + ')';
    const aw = $('airway');
    aw.classList.toggle('critical', patency < 25 && patency > 0);
    aw.classList.toggle('closed', patency <= 0);
}

function addStream(v, conf, lat, saved) {
    const el = document.createElement('div');
    el.className = 'decision ' + (saved ? 'saved' : 'tooslow');
    el.innerHTML = `<span class="d-left"><span class="d-sw" style="background:${v.color}"></span>
      <span class="d-name">${v.name}</span><span class="d-act">IM Epinephrine 0.3 mg</span></span>
    <span class="d-right">${conf != null ? `<span class="d-conf">${conf.toFixed(2)}</span>` : ''}
      <span class="d-lat">${Math.round(lat)}ms</span><span class="d-out">${saved ? 'SAVED' : 'CODE BLUE'}</span></span>`;
    const stream = $('stream');
    stream.prepend(el);
    while (stream.children.length > 50) stream.removeChild(stream.lastChild);
}

function updateStats() {
    const saved = variants.filter(v => resolved[v.id] && $('racer-' + v.id).classList.contains('saved')).length;
    $('stPatients').textContent = patients;
    $('stSaved').textContent = saved + '/' + variants.length;
    $('stCode').textContent = (variants.length - saved) + '/' + variants.length;
    $('stFastest').textContent = variants.length ? fmtMs(variants[0].mean_latency_ms) : '—';
}

// ---------- controls ----------
function togglePlay() {
    if (playing) { playing = false; cancelAnimationFrame(raf); $('play').textContent = '▶ Play'; return; }
    playing = true; $('play').textContent = 'Ⅱ Pause';
    last = performance.now();
    raf = requestAnimationFrame(tick);
}

$('play').onclick = togglePlay;
$('reset').onclick = () => { playing = false; cancelAnimationFrame(raf); $('play').textContent = '▶ Play'; patients = 0; $('stream').innerHTML = ''; newPatient(); };
$('closure').oninput = e => {
    closureMs = Number(e.target.value);
    $('closureVal').textContent = closureMs + ' ms';
    if (!playing) { caseDur = closureMs * TIME_SCALE / mult; renderAirway(0); }
};
$('mult').onchange = e => { mult = Number(e.target.value); };
document.addEventListener('keydown', e => { if (e.code === 'Space') { e.preventDefault(); togglePlay(); } });

load();
