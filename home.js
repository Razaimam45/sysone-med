// Populate the briefing with the real per-variant measurements from /api/snake.
const $ = id => document.getElementById(id);
const fmtMs = ms => ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : Math.round(ms) + ' ms';

async function init() {
  let cfg;
  try {
    const r = await fetch('data/snake.json');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    cfg = await r.json();
  } catch (e) {
    $('variantGrid').innerHTML =
      '<p class="muted">Could not load variant data: ' + e.message +
      '. Make sure demo/data/comparison.json exists (run <code>python demo/server.py</code> once).</p>';
    return;
  }
  const vs = cfg.variants; // already sorted fastest -> slowest by mean latency

  // Stat strip.
  const fastest = vs[0], slowest = vs[vs.length - 1];
  $('statFastest').textContent = fmtMs(fastest.mean_latency_ms);
  $('statSlowest').textContent = fmtMs(slowest.mean_latency_ms);
  $('statSpread').textContent = (slowest.mean_latency_ms / fastest.mean_latency_ms).toFixed(0) + '×';
  // Best SLAKE accuracy: recompute from the VQA bundle for the SLAKE subset.
  bestSlakeAccuracy().then(a => { $('statAcc').textContent = a == null ? '—' : a.toFixed(1) + '%'; });

  // Variant cards.
  $('variantGrid').innerHTML = vs.map((v, i) => `
    <article class="vcard" style="--c:${v.color}">
      <span class="rank">SPEED #${i + 1}</span>
      <span class="dot"></span>
      <div class="vname">${v.name}</div>
      <span class="vtag">${v.tag}</span>
      <p class="vblurb">${v.blurb}</p>
      <div class="vmetrics">
        <div><strong>${fmtMs(v.mean_latency_ms)}</strong><small>MEAN DECISION</small></div>
        <div><strong>${v.accuracy == null ? '—' : v.accuracy.toFixed(1) + '%'}</strong><small>HELD-OUT ACC</small></div>
      </div>
    </article>`).join('');
}

// Best held-out accuracy on SLAKE across variants (the visual task).
async function bestSlakeAccuracy() {
  try {
    const r = await fetch('data/slake.json');
    if (!r.ok) return null;
    const b = await r.json();
    let best = null;
    for (const m of b.models) {
      const recs = Object.values(b.recorded[m.id]);
      if (!recs.length) continue;
      const acc = recs.reduce((s, x) => s + (x.correct ? 1 : 0), 0) / recs.length * 100;
      if (best == null || acc > best) best = acc;
    }
    return best;
  } catch { return null; }
}

init();
