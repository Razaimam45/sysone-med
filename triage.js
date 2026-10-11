// SysOne-Med · Time is Brain — rapid triage race.
// A critical case appears with a closing treatment window. All five decision
// paths race to make the right call before the window shuts. Each path takes
// its MEASURED mean latency to decide: if it finishes before the window closes
// it saves the patient, if it's too slow the window shuts first. Speed is the
// whole game — accuracy is not required to win the race. The decision stream
// logs every call: action, confidence, latency, and outcome.
const $ = id => document.getElementById(id);
const TIME_SCALE = 3;   // visual ms per real ms, so a 400ms window is watchable

let variants = [];      // sorted fastest -> slowest
let windowMs = 400;
let mult = 0.5;
let cases = 0, raf = null, last = 0, playing = false;
let caseData = null, caseStart = 0, caseDur = 0, resolved = {};

const clamp01 = x => Math.max(0.02, Math.min(0.999, x));
const fmtMs = ms => ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : Math.round(ms) + ' ms';

// ---------- medical scenarios ----------
const SCENARIOS = [
  { tag: 'STROKE · LVO', state: () => `Onset ${ri(20, 90)} min · NIHSS ${ri(8, 22)} · CTA: LVO + · BP ${ri(150, 210)}/${ri(80, 110)}`,
    opts: ['Administer Tenecteplase (tPA)', 'Clear for Mechanical Thrombectomy', 'Hold meds & lower BP', 'Intubate & transfer to ICU'], correct: 1 },
  { tag: 'ANAPHYLAXIS', state: () => `Stridor ${ri(70, 95)} dB · SpO₂ ${ri(78, 92)}% · Tongue swelling ${ri(3, 5)}/5 · HR ${ri(110, 150)}`,
    opts: ['IM Epinephrine 0.3 mg', 'Nebulized Albuterol', 'IV Diphenhydramine', 'Prepare Cricothyroidotomy'], correct: 0 },
  { tag: 'CARDIAC ARREST', state: () => `Rhythm: Asystole · HR ${ri(0, 10)} · SpO₂ ${ri(60, 80)}% · ROSC: none · ${ri(1, 6)} min down`,
    opts: ['Start CPR + Epinephrine', 'Defibrillate', 'IV Atropine', 'Observe & reassess'], correct: 0 },
  { tag: 'SEPSIS', state: () => `qSOFA ${ri(2, 3)}/3 · HR ${ri(100, 140)} · MAP ${ri(55, 70)} · Lactate ${rf(3, 8)} mmol/L · Temp ${rf(37.5, 39.5)}°C`,
    opts: ['Broad-spectrum antibiotics + fluids', 'Blood cultures only', 'IV Steroids', 'Discharge with follow-up'], correct: 0 },
  { tag: 'HYPERTENSIVE EMERG', state: () => `BP ${ri(210, 260)}/${ri(120, 150)} · End-organ: ${pick(['encephalopathy', 'pulmonary edema', 'aortic dissection'])} · HR ${ri(80, 110)}`,
    opts: ['IV antihypertensive (nicardipine)', 'Oral antihypertensive', 'Discharge', 'IV fluids'], correct: 0 },
  { tag: 'MASSIVE PE', state: () => `RV strain: + · SpO₂ ${ri(80, 90)}% · BP ${ri(80, 100)}/${ri(50, 70)} · D-dimer ${ri(8, 20)}× ULN`,
    opts: ['Systemic thrombolysis', 'Anticoagulation only', 'Observation', 'IV fluids'], correct: 0 },
  { tag: 'DKA', state: () => `Glucose ${ri(400, 700)} mg/dL · pH ${rf(6.9, 7.2)} · HCO₃ ${ri(8, 16)} · K⁺ ${rf(3.5, 5.5)} mEq/L`,
    opts: ['IV insulin + fluids', 'Oral insulin', 'IV Bicarbonate', 'Discharge'], correct: 0 },
  { tag: 'TENSION PNEUMOTHORAX', state: () => `Tracheal deviation: + · SpO₂ ${ri(75, 88)}% · BP ${ri(80, 100)}/${ri(50, 70)} · HR ${ri(120, 160)}`,
    opts: ['Needle decompression', 'Chest tube', 'Observation', 'Intubate'], correct: 0 },
];
const ri = (a, b) => a + Math.floor(Math.random() * (b - a + 1));
const rf = (a, b) => (a + Math.random() * (b - a)).toFixed(1);
const pick = arr => arr[Math.floor(Math.random() * arr.length)];

function nextCase() {
  const s = SCENARIOS[Math.floor(Math.random() * SCENARIOS.length)];
  return { tag: s.tag, state: s.state(), opts: s.opts, correct: s.correct };
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
  newCase();
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

// ---------- case lifecycle ----------
function newCase() {
  caseData = nextCase();
  cases++;
  resolved = {};
  $('caseTag').textContent = caseData.tag;
  $('caseNum').textContent = 'CASE ' + String(cases).padStart(4, '0');
  $('caseState').textContent = caseData.state;
  $('caseOpts').innerHTML = caseData.opts.map((o, i) =>
    `<div class="case-opt ${i === caseData.correct ? 'correct' : ''}"><span class="k">${String.fromCharCode(65 + i)}</span>${o}</div>`).join('');
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
  // visual duration of this case
  caseDur = windowMs * TIME_SCALE / mult;
  caseStart = performance.now();
  if (playing) { last = caseStart; raf = requestAnimationFrame(tick); }
  else { renderWindow(0); }
}

function tick(now) {
  if (!playing) return;
  const elapsed = now - caseStart;
  const t = Math.min(1, elapsed / caseDur);
  renderWindow(t);
  // resolve each racer as its (scaled) latency is reached
  for (const v of variants) {
    if (resolved[v.id]) continue;
    const frac = v.mean_latency_ms / windowMs;      // where it finishes relative to window
    if (frac >= 1) {
      // too slow: resolves exactly when the window closes
      if (t >= 1) resolveRacer(v, false, now);
    } else if (t >= frac) {
      resolveRacer(v, true, now);
    }
  }
  if (t >= 1) {
    // window closed — any unresolved racer is too slow
    for (const v of variants) if (!resolved[v.id]) resolveRacer(v, false, now);
    // brief pause, then next case
    setTimeout(() => { if (playing) newCase(); }, 900 / mult);
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
  verdict.textContent = saved ? '✓ SAVED' : '✕ TOO SLOW';
  verdict.className = 'r-verdict ' + (saved ? 'saved' : 'tooslow');
  el.querySelector('.r-bar i').style.width = (saved ? Math.min(100, lat / windowMs * 100) : 100) + '%';
  el.querySelector('.r-conf').textContent = conf != null ? conf.toFixed(2) : '—';
  el.querySelector('.r-out').textContent = Math.round(lat) + ' ms';
  addStream(v, caseData.opts[caseData.correct], conf, lat, saved);
  updateStats();
}

function renderWindow(t) {
  $('windowFill').style.width = ((1 - t) * 100) + '%';
  $('windowMs').textContent = Math.max(0, Math.round(windowMs * (1 - t))) + ' ms';
}

function addStream(v, action, conf, lat, saved) {
  const el = document.createElement('div');
  el.className = 'decision ' + (saved ? 'saved' : 'tooslow');
  el.innerHTML = `<span class="d-left"><span class="d-sw" style="background:${v.color}"></span>
      <span class="d-name">${v.name}</span><span class="d-act">${action}</span></span>
    <span class="d-right">${conf != null ? `<span class="d-conf">${conf.toFixed(2)}</span>` : ''}
      <span class="d-lat">${Math.round(lat)}ms</span><span class="d-out">${saved ? 'SAVED' : 'TOO SLOW'}</span></span>`;
  const stream = $('stream');
  stream.prepend(el);
  while (stream.children.length > 50) stream.removeChild(stream.lastChild);
}

function updateStats() {
  const saved = variants.filter(v => resolved[v.id] && $('racer-' + v.id).classList.contains('saved')).length;
  $('stCases').textContent = cases;
  $('stSaved').textContent = saved + '/' + variants.length;
  $('stFastest').textContent = variants.length ? fmtMs(variants[0].mean_latency_ms) : '—';
  $('stWindow').textContent = windowMs + ' ms';
}

// ---------- controls ----------
function togglePlay() {
  if (playing) { playing = false; cancelAnimationFrame(raf); $('play').textContent = '▶ Play'; return; }
  playing = true; $('play').textContent = 'Ⅱ Pause';
  last = performance.now();
  raf = requestAnimationFrame(tick);
}

$('play').onclick = togglePlay;
$('reset').onclick = () => { playing = false; cancelAnimationFrame(raf); $('play').textContent = '▶ Play'; cases = 0; $('stream').innerHTML = ''; newCase(); };
$('window').oninput = e => {
  windowMs = Number(e.target.value);
  $('windowVal').textContent = windowMs + ' ms';
  $('stWindow').textContent = windowMs + ' ms';
  if (!playing) { caseDur = windowMs * TIME_SCALE / mult; renderWindow(0); }
};
$('mult').onchange = e => { mult = Number(e.target.value); };
document.addEventListener('keydown', e => { if (e.code === 'Space') { e.preventDefault(); togglePlay(); } });

load();
