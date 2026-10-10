// SysOne-Med · Snake race — five simultaneous panels.
// Each panel is an independent race: one snake, its own apples, its own timer.
// A snake makes a move per "decision"; a decision costs that variant's measured
// mean latency, so faster variants outrun slower ones. All five run on one shared
// clock so you can compare speed side by side. On-screen speed is rescaled into a
// readable range (see /api/snake); the leaderboard shows the real ms values.
const $ = id => document.getElementById(id);
const DIRS = { up: {x:0,y:-1}, down:{x:0,y:1}, left:{x:-1,y:0}, right:{x:1,y:0} };
const TURN = { up:'right', right:'down', down:'left', left:'up' };
const TURN2 = { up:'left', left:'down', down:'right', right:'up' };
const OPP = { up:'down', down:'up', left:'right', right:'left' };

// Fixed panel order (not by speed): MedJev, AR-Gen, AR-LL, AR-Head, NAR.
const PANEL_ORDER = ['medjev', 'gen', 'll', 'head', 'nar'];

const COLS = 15, ROWS = 11, CELL = 24;
let cfg = null, panels = [], playing = false, raceTime = 0, target = 5, mult = 1, raf = null, last = 0;
let firstWinner = null;

// Read the user's apple target from the input, clamped to a sane range.
function readTarget() {
  const el = $('apples');
  let n = parseInt(el ? el.value : '5', 10);
  if (isNaN(n) || n < 1) n = 1;
  if (n > 20) n = 20;
  if (el) el.value = n;
  return n;
}

const key = (x, y) => x + ',' + y;
const inBounds = (x, y) => x >= 0 && y >= 0 && x < COLS && y < ROWS;
const manhattan = (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);

function makePanel(v, i) {
  const canvas = document.createElement('canvas');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = COLS * CELL * dpr;
  canvas.height = ROWS * CELL * dpr;
  canvas.style.aspectRatio = COLS + ' / ' + ROWS;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const y = 1 + (i % ROWS);
  return {
    v, canvas, ctx,
    snake: { body: [{x:2,y},{x:1,y},{x:0,y}], dir:'right', apples:0, acc:0,
             interval: 1000 / v.on_screen_moves_per_sec },
    apples: new Set(),
    won: false, dead: false, wonTime: null,
  };
}

function spawnApple(p) {
  const free = [];
  for (let y = 0; y < ROWS; y++) for (let x = 0; x < COLS; x++) {
    if (p.apples.has(key(x, y))) continue;
    if (p.snake.body.some(c => c.x === x && c.y === y)) continue;
    free.push({x, y});
  }
  if (!free.length) return;
  const a = free[Math.floor(Math.random() * free.length)];
  p.apples.add(key(a.x, a.y));
}

function nearestApple(p, head) {
  let best = null, bd = Infinity;
  for (const k of p.apples) {
    const [x, y] = k.split(',').map(Number);
    const d = manhattan(head, {x, y});
    if (d < bd) { bd = d; best = {x, y}; }
  }
  return best;
}

function step(p) {
  const s = p.snake, head = s.body[0];
  const apple = nearestApple(p, head);
  const cands = [s.dir, TURN[s.dir], TURN2[s.dir]].filter(d => d !== OPP[s.dir]);
  let best = null, bestScore = Infinity;
  for (const d of cands) {
    const v = DIRS[d], nx = head.x + v.x, ny = head.y + v.y;
    if (!inBounds(nx, ny)) continue;
    const willGrow = apple && nx === apple.x && ny === apple.y;
    const body = willGrow ? s.body : s.body.slice(0, -1);
    if (body.some(c => c.x === nx && c.y === ny)) continue;
    const score = apple ? manhattan({x:nx,y:ny}, apple) : 0;
    if (score < bestScore) { bestScore = score; best = {d, nx, ny, willGrow}; }
  }
  if (!best) { p.dead = true; return; }
  s.dir = best.d;
  s.body.unshift({x: best.nx, y: best.ny});
  if (best.willGrow) {
    s.apples++;
    p.apples.delete(key(best.nx, best.ny));
    spawnApple(p);
    if (s.apples >= target) { p.won = true; p.wonTime = raceTime; if (!firstWinner) firstWinner = p; }
  } else {
    s.body.pop();
  }
}

function frame(now) {
  if (!playing) return;
  const dt = Math.min(100, now - last); last = now;
  raceTime += dt / 1000;
  for (const p of panels) {
    if (p.won || p.dead) continue;
    p.snake.acc += dt * mult;
    while (p.snake.acc >= p.snake.interval && !p.won && !p.dead) {
      p.snake.acc -= p.snake.interval;
      step(p);
    }
  }
  $('clock').textContent = raceTime.toFixed(1);
  renderAll();
  const done = panels.every(p => p.won || p.dead);
  if (done) { playing = false; $('play').textContent = '▶ Start race'; setStatus('Race complete.'); }
  if (playing) raf = requestAnimationFrame(frame);
}

function start() {
  if (panels.every(p => p.won || p.dead)) reset();
  if (playing) { playing = false; $('play').textContent = '▶ Resume'; return; }
  playing = true; $('play').textContent = 'Ⅱ Pause'; setStatus('Racing… all five at once.');
  last = performance.now();
  raf = requestAnimationFrame(frame);
}

function reset() {
  if (!cfg) return;
  target = readTarget();
  $('targetLabel').textContent = target;
  $('targetLabel2').textContent = target;
  // Fixed panel order: MedJev, AR-Gen, AR-LL, AR-Head, NAR.
  const ordered = PANEL_ORDER
    .map(id => cfg.variants.find(v => v.id === id))
    .filter(Boolean);
  panels = ordered.map((v, i) => makePanel(v, i));
  // One apple per panel at a time; the next appears only after it is eaten.
  for (const p of panels) spawnApple(p);
  raceTime = 0; playing = false; firstWinner = null;
  $('play').textContent = '▶ Start race';
  $('clock').textContent = '0.0';
  setStatus('Ready — press Start. All five races run simultaneously.');
  buildPanels(); renderAll();
}

function setStatus(t) { $('statusText').textContent = t; }

// ---------- DOM ----------
function buildPanels() {
  const grid = $('raceGrid');
  grid.innerHTML = '';
  for (const p of panels) {
    const el = document.createElement('div');
    el.className = 'race-panel';
    el.style.setProperty('--c', p.v.color);
    el.innerHTML = `
      <div class="rp-head"><span class="sw" style="background:${p.v.color}"></span>
        <span class="nm">${p.v.name}</span><span class="ms">${p.v.mean_latency_ms} ms</span></div>
      <div class="rp-canvas">${p.canvas.outerHTML}</div>
      <div class="rp-foot"><span class="ap"><b class="apn">0</b> / ${target}</span>
        <span class="mps">${(1000/p.snake.interval).toFixed(1)} moves/s</span></div>
      <span class="rp-badge"></span>`;
    grid.appendChild(el);
    // re-grab the canvas element that outerHTML cloned
    p.canvas = el.querySelector('canvas');
    const dpr = window.devicePixelRatio || 1;
    p.canvas.width = COLS * CELL * dpr; p.canvas.height = ROWS * CELL * dpr;
    p.ctx = p.canvas.getContext('2d'); p.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    p.el = el;
  }
}

function renderAll() {
  const leader = panels.reduce((a, b) => (b.snake.apples > a.snake.apples ? b : a), panels[0]);
  for (const p of panels) {
    p.el.classList.toggle('leading', !p.won && !p.dead && p === leader);
    p.el.classList.toggle('dead', p.dead);
    p.el.querySelector('.apn').textContent = p.snake.apples;
    const badge = p.el.querySelector('.rp-badge');
    if (p.won) { badge.textContent = '🏆 WIN'; badge.classList.add('win'); }
    else if (p.dead) { badge.textContent = 'CRASHED'; badge.classList.remove('win'); }
    else { badge.textContent = ''; badge.classList.remove('win'); }
    draw(p);
  }
  renderLeaderboard();
}

function renderLeaderboard() {
  const sorted = [...panels].sort((a, b) => b.snake.apples - a.snake.apples || a.snake.interval - b.snake.interval);
  $('leaderboard').innerHTML = sorted.map(p => `
    <div class="lb-row ${p.dead ? 'dead' : ''}">
      <span class="sw" style="background:${p.v.color}"></span>
      <span class="nm">${p.v.name}<small>${p.v.mean_latency_ms} ms/decision · ${(1000/p.snake.interval).toFixed(1)} moves/s</small></span>
      <span class="bar"><i style="width:${Math.min(100, p.snake.apples/target*100)}%;background:${p.v.color}"></i></span>
      <span class="ap">${p.snake.apples}</span>
    </div>`).join('');
}

// ---------- drawing ----------
function draw(p) {
  const {ctx} = p, W = COLS * CELL, H = ROWS * CELL;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(255,255,255,0.04)'; ctx.lineWidth = 1;
  for (let x = 0; x <= COLS; x++) { ctx.beginPath(); ctx.moveTo(x*CELL,0); ctx.lineTo(x*CELL,H); ctx.stroke(); }
  for (let y = 0; y <= ROWS; y++) { ctx.beginPath(); ctx.moveTo(0,y*CELL); ctx.lineTo(W,y*CELL); ctx.stroke(); }
  for (const k of p.apples) {
    const [x, y] = k.split(',').map(Number);
    ctx.fillStyle = '#e0563f';
    ctx.beginPath(); ctx.arc(x*CELL+CELL/2, y*CELL+CELL/2, CELL*0.28, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#7bbf6a'; ctx.fillRect(x*CELL+CELL/2-1, y*CELL+CELL*0.18, 2, CELL*0.14);
  }
  const s = p.snake, alpha = (p.won || p.dead) ? 0.5 : 1;
  s.body.forEach((c, i) => {
    const isHead = i === 0;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = isHead ? p.v.color : shade(p.v.color, i * 0.06);
    roundRect(ctx, c.x*CELL+2, c.y*CELL+2, CELL-4, CELL-4, isHead ? 8 : 5);
    ctx.fill();
    if (isHead) {
      ctx.fillStyle = '#0d161c';
      const ex = c.x*CELL+CELL/2, ey = c.y*CELL+CELL/2;
      ctx.beginPath(); ctx.arc(ex-4, ey-2, 2, 0, Math.PI*2); ctx.arc(ex+4, ey-2, 2, 0, Math.PI*2); ctx.fill();
    }
  });
  ctx.globalAlpha = 1;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x+r, y);
  ctx.arcTo(x+w, y, x+w, y+h, r);
  ctx.arcTo(x+w, y+h, x, y+h, r);
  ctx.arcTo(x, y+h, x, y, r);
  ctx.arcTo(x, y, x+w, y, r);
  ctx.closePath();
}

function shade(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  let r = (n>>16)&255, g=(n>>8)&255, b=n&255;
  r = Math.max(0, Math.min(255, r - amt*255));
  g = Math.max(0, Math.min(255, g - amt*255));
  b = Math.max(0, Math.min(255, b - amt*255));
  return `rgb(${r|0},${g|0},${b|0})`;
}

// ---------- boot ----------
async function boot() {
  try {
    const r = await fetch('/api/snake');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    cfg = await r.json();
  } catch (e) { setStatus('Could not load race data: ' + e.message); return; }
  reset();
  $('play').onclick = start;
  $('reset').onclick = reset;
  $('mult').onchange = e => { mult = Number(e.target.value); };
  $('apples').onchange = () => { if (!playing) reset(); };
}
boot();
