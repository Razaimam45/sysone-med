const $ = id => document.getElementById(id);
const colors = {medjev:'#a17bd3',gen:'#d69957',ll:'#4fc3c9',head:'#658ed0',nar:'#48a895'};
const pct = x => x == null ? '—' : (x * 100).toFixed(1) + '%';
const num = x => x == null ? '—' : x.toFixed(1);
const esc = x => String(x).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data, step = 0, playing = false, timer, dataset = 'slake', loadVersion = 0;
let history = {}, summaries = {}, timeline = {}, counts = {}, elapsed = 0, lastTick = 0, previewClock = 0;
let currentImage = null, lastInput = null, busy = true;
const imageCache = new Map();
const isRace = () => true; // Always in race mode now
const isLiveInference = () => $('inferenceMode').value === 'live';

// Run live inference for a sample and model
async function runLiveInference(sample, modelKey) {
  const response = await fetch(`/api/inference?dataset=${dataset}&model=${modelKey}&sampleId=${encodeURIComponent(sample.id)}`);
  if (!response.ok) throw new Error(`HTTP ${response.status} ${await response.text()}`);
  return await response.json();
}
const sampleAt = i => data.samples[i % data.unique];
// Live results are filled in by workers as each model finishes a case.
let liveResults = {}, runToken = 0, rendering = false, renderAgain = false;
const liveCache = new Map(), workerRunning = new Set();
function predictionAt(i, key) {
  if (isLiveInference()) return liveResults[key][i];
  return data.recorded[key][sampleAt(i).id];
}

async function scheduleRender() {
  if (rendering) { renderAgain = true; return; }
  rendering = true;
  try { do { renderAgain = false; await render(); } while (renderAgain); }
  finally { rendering = false; }
}

async function liveWorker(m, token) {
  if (workerRunning.has(m.id + token)) return;
  workerRunning.add(m.id + token);
  try {
    while (playing && token === runToken && counts[m.id] < data.total) {
      const i = counts[m.id], sample = sampleAt(i), cacheKey = m.id + '|' + sample.id;
      let result = liveCache.get(cacheKey);
      if (!result) {
        result = await runLiveInference(sample, m.id);
        liveCache.set(cacheKey, result);
      }
      if (token !== runToken) return;
      liveResults[m.id][i] = result; timeline[m.id][i] = elapsed; counts[m.id] = i + 1;
      scheduleRender();
    }
  } catch (error) {
    if (token === runToken) { stop(); $('runStatus').textContent = `Live inference failed for ${m.name}: ${error.message}`; }
  } finally { workerRunning.delete(m.id + token); }
}

function resetRace() {
  runToken++;
  elapsed = 0;
  counts = Object.fromEntries(data.models.map(m => [m.id, 0]));
  liveResults = Object.fromEntries(data.models.map(m => [m.id, []]));
  timeline = Object.fromEntries(data.models.map(m => [m.id, isLiveInference() ? [] : RaceTiming.completionTimes(data, m.id)]));
}

async function preloadImages(samples, version) {
  const urls = [...new Set(samples.map(s => s.image_url).filter(Boolean))];
  let next = 0, loaded = 0;
  async function worker() {
    while (next < urls.length && version === loadVersion) {
      const url = urls[next++];
      if (!imageCache.has(url)) {
        const image = new Image();
        image.src = url;
        try { await image.decode(); imageCache.set(url, image); }
        catch { imageCache.set(url, null); }
      }
      loaded++;
      if (version === loadVersion) $('runStatus').textContent = `Preparing scans ${loaded}/${urls.length}…`;
    }
  }
  await Promise.all(Array.from({length: Math.min(8, urls.length)}, worker));
}

function updateImage(sample) {
  const identity = sample.image_url || sample.id;
  if (currentImage === identity) return;
  currentImage = identity;
  const image = imageCache.get(sample.image_url);
  if (image) {
    // Insert the already-decoded element itself: no request or decode per frame.
    const old = $('scan'); old.removeAttribute('id');
    image.id = 'scan'; image.alt = `SLAKE scan ${sample.image_id} for: ${sample.question}`;
    image.hidden = false; old.replaceWith(image); $('imageError').hidden = true;
  } else {
    $('scan').hidden = true; $('imageError').hidden = false;
    $('imageError').textContent = sample.image_url ? 'Scan unavailable' :
      sample.context || 'Read the clinical vignette alongside. No image is provided for this task.';
  }
}

function renderInput() {
  const sample = sampleAt(step);
  if (lastInput === dataset + ':' + step) return;
  lastInput = dataset + ':' + step;
  $('seek').max = data.total - 1; $('seek').value = step;
  $('position').textContent = `${isRace() ? 'INSPECTING' : 'SAMPLE'} ${String(step+1).padStart(4,'0')} / ${data.total.toLocaleString()}`;
  $('sampleId').textContent = sample.id;
  $('cycle').textContent = step >= data.unique ? `Repeat · pass ${Math.floor(step/data.unique)+1}` : 'First pass';
  $('question').textContent = sample.question; $('optionCount').textContent = `${sample.options.length} choices`;
  $('options').innerHTML = sample.options.map((o,i) => `<div class="option ${i===sample.label?'truth':''}"><b>${String.fromCharCode(65+i)}</b>${esc(o)}</div>`).join('');
  updateImage(sample);
  $('imageIdentity').textContent = sample.image_id ? 'SCAN / '+sample.image_id : 'TEXT-ONLY INPUT';
  $('nextImage').hidden = !sample.image_url;
  $('scanLabel').textContent = sample.image_url ? 'MEDICAL IMAGE' : 'CLINICAL CONTEXT';
}

async function aggregate() {
  history = {}; summaries = {};
  for (const m of data.models) {
    const limit = isRace() ? counts[m.id] : step + 1;
    let correct = 0, confidence = 0, cn = 0, brier = 0, lat = 0;
    const lats = [], bins = Array.from({length:10}, () => [0,0,0]);
    let s = {n:0, accuracy:null, confidence:null, brier:null, latency:null, ece:null};
    history[m.id] = [];
    for (let i=0; i<limit; i++) {
      const r = await predictionAt(i,m.id);
      const n = i+1;
      correct += r.correct; brier += r.brier;
      if (r.confidence != null) {
        confidence += r.confidence; cn++;
        const bin = bins[Math.min(9,Math.floor(r.confidence*10))];
        bin[0]++; bin[1] += r.confidence; bin[2] += r.correct;
      }
      lat += r.latency_ms; lats.push(r.latency_ms);
      s = {n, accuracy:correct/n, confidence:cn?confidence/cn:null, brier:brier/n,
        latency:lat/n, ece:cn?bins.reduce((v,b)=>v+Math.abs(b[1]-b[2]),0)/cn:null};
      history[m.id].push({...s});
    }
    lats.sort((a,b)=>a-b);
    summaries[m.id] = {...s, p95:lats.length?lats[Math.ceil(.95*lats.length)-1]:null};
  }
}

async function renderCards() {
  $('cards').style.gridTemplateColumns = innerWidth > 1000 ? `repeat(${data.models.length},1fr)` : '';
  const cards = [];
  for (const m of data.models) {
    const index = isRace() ? counts[m.id]-1 : step;
    const r = index >= 0 ? await predictionAt(index,m.id) : null;
    const sample = index >= 0 ? sampleAt(index) : null;
    const answer = r ? sample.options[r.prediction] || 'Invalid answer' : 'Awaiting first decision';
    const total = isRace() ? counts[m.id] : step+1;
    cards.push(`<article class="model-card" style="--color:${colors[m.id]}">
      <div class="model-heading">${m.name}<span class="verdict ${!r?'missing':r.correct?'':'wrong'}">${!r?'RUNNING':r.correct?'CORRECT':'INCORRECT'}</span></div>
      <div class="model-sub">${m.subtitle}</div>
      <div class="model-count"><strong>${total.toLocaleString()}</strong><span> / ${data.total.toLocaleString()} scored</span></div>
      ${isRace() ? `<div class="race-track"><i style="width:${total/data.total*100}%"></i></div><button class="inspect-model" data-model="${m.id}" ${index<0?'disabled':''}>${index<0?'No completed sample':`Inspect latest · #${index+1} ↗`}</button>` : ''}
      <div class="prediction" title="${esc(sample ? sample.question+' → '+answer : answer)}">${esc(answer)}</div>
      <div class="card-stats"><div><strong>${r?num(r.latency_ms):'—'}</strong><small>LATENCY / MS</small></div><div><strong>${r?pct(r.confidence):'—'}</strong><small>CONFIDENCE</small></div></div>
      <div class="confidence-track"><div style="width:${(r?.confidence||0)*100}%"></div></div></article>`);
  }
  $('cards').innerHTML = cards.join('');
}

function mechanismSteps(id) {
  // Naive, plain-language walkthrough of how each method turns an input into a decision.
  // Based on the actual adapters used to collect these predictions.
  const steps = {
    gen: [
      'Give the question and every answer choice to MedGemma as a chat prompt.',
      'Autoregressive: let the model write free text, up to 32 tokens, one token at a time.',
      'Parse the first written tokens and match them to a choice ("A", "A.", "A)", "The answer is A").',
      'If nothing matches, the sample counts as invalid (wrong).',
      'Confidence = the probability the model gave the token it actually emitted.'
    ],
    ll: [
      'Give the same question + choices prompt to MedGemma.',
      'Instead of writing many tokens, let the model emit exactly one token — a single next-word guess.',
      'If that one token is a valid choice letter (A/B/C/D) it is the answer; otherwise the sample is invalid (wrong).',
      'Producing just one token makes this the fastest of the two AR paths.'
    ],
    head: [
      'Feed the input (clinical text, or text + SLAKE image) into a frozen MedGemma backbone — no token-by-token writing.',
      'A small trained pointer head scores each answer option against a special <decide> position.',
      'Softmax turns the scores into probabilities; the highest-scoring option is the answer.',
      'Only the decision head is trained; the backbone stays frozen for speed.'
    ],
    nar: [
      'Encode the clinical text and each answer option with a text encoder (plus a vision encoder when an image is present, e.g. SLAKE).',
      'A trained fusion layer combines text and image features.',
      'A calibrated decision head scores all options directly in one forward pass — no generation.',
      'Softmax converts the scores into calibrated probabilities; the top option wins.'
    ]
  };
  return steps[id] || [];
}

const MECHANISM_META = {
  gen: 'Autoregressive · generative',
  ll: 'Autoregressive · single token',
  head: 'Classifier head · frozen backbone',
  nar: 'Non-autoregressive'
};

function mechanismSVG(id, color) {
  // Vertical timeline diagram showing how each method reaches a decision.
  const boxes = ({
    gen: ['Question + choices', 'Write up to 32 tokens', 'Parse the text', 'Match A / B / C / D'],
    ll: ['Question + choices', 'Emit exactly 1 token', 'Is it A / B / C / D?'],
    head: ['Input text (+ image)', 'Frozen MedGemma backbone', 'Pointer head scores options', 'Softmax → best option'],
    nar: ['Input text + image', 'Text + vision encoders', 'Fusion + decision head', 'Softmax → best option']
  })[id] || [];
  if (!boxes.length) return '';
  const w = 300, bh = 34, gap = 20, top = 8;
  const railX = 15, boxX = 36, boxW = w - boxX - 4;
  const h = boxes.length * bh + (boxes.length - 1) * gap + 2 * top;
  const firstCy = top + bh / 2, lastCy = top + (boxes.length - 1) * (bh + gap) + bh / 2;
  const rail = `<line x1="${railX}" y1="${firstCy}" x2="${railX}" y2="${lastCy}" stroke="${color}" stroke-opacity=".3" stroke-width="2"/>`;
  const marks = boxes.map((label, i) => {
    const y = top + i * (bh + gap);
    const cy = y + bh / 2;
    return [
      `<rect x="${boxX}" y="${y}" width="${boxW}" height="${bh}" rx="9" fill="${color}" fill-opacity=".07"/>`,
      `<rect x="${boxX}" y="${y}" width="${boxW}" height="${bh}" rx="9" fill="none" stroke="${color}" stroke-opacity=".45" stroke-width="1.2"/>`,
      `<text x="${boxX + boxW / 2}" y="${cy + 4}" text-anchor="middle" font-size="11.5" font-weight="500" font-family="Inter,ui-sans-serif,system-ui,sans-serif" fill="#202634">${esc(label)}</text>`,
      `<circle cx="${railX}" cy="${cy}" r="9" fill="${color}"/>`,
      `<text x="${railX}" y="${cy + 3.5}" text-anchor="middle" font-size="10" font-weight="700" font-family="Inter,ui-sans-serif,system-ui,sans-serif" fill="#ffffff">${i + 1}</text>`
    ].join('');
  }).join('');
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(id)} inference flow" style="width:100%;height:auto;display:block">${rail}${marks}</svg>`;
}

function renderMechanism() {
  const panel = $('mechanism');
  if (!panel) return;
  const cards = data.models.map(m => {
    const color = colors[m.id];
    if (m.id === 'medjev') {
      // MedJev is explained by its own repository — just redirect to GitHub.
      return `<article class="mechanism-card github-card" style="--color:${color}">
        <div class="mc-head"><span class="mc-dot"></span><span class="mc-name">${esc(m.name)}</span></div>
        <div class="mc-chip">External model</div>
        <p class="mc-note">MedJev is a separate non-autoregressive system with its own repository. Its decision mechanism, training details, and results live in the project documentation rather than this comparison.</p>
        <a class="github-cta" href="https://github.com/JunMa11/MedJev" target="_blank" rel="noopener" aria-label="Open MedJev on GitHub">View MedJev on GitHub<span class="github-arrow">↗</span></a>
      </article>`;
    }
    const svg = mechanismSVG(m.id, color);
    const steps = mechanismSteps(m.id);
    return `<article class="mechanism-card" style="--color:${color}">
      <div class="mc-head"><span class="mc-dot"></span><span class="mc-name">${esc(m.name)}</span></div>
      <div class="mc-chip">${esc(MECHANISM_META[m.id] || m.subtitle)}</div>
      <p class="mc-note">${esc(m.subtitle)}</p>
      ${svg}
      <ol class="mc-steps">${steps.map(s => '<li>' + esc(s) + '</li>').join('')}</ol>
    </article>`;
  }).join('');
  panel.innerHTML = `<div class="mechanism-grid">${cards}</div>`;
}

async function render() {
  if (!data || busy) return;
  renderInput();
  await aggregate();
  await renderCards();
  renderMechanism();
  const progress = isRace() ? Math.min(...Object.values(counts))/data.total : (step+1)/data.total;
  $('overviewProgress').textContent = isRace() ? (elapsed/1000).toFixed(1)+' s' : pct(progress);
  $('progress').style.width = progress*100+'%';
  $('overviewPass').textContent = isRace() ? 'Shared replay clock · independent workers' : $('cycle').textContent;
  $('runStatus').textContent = isLiveInference() ? 'Live inference · each model runs on the server independently' : 'Saved-latency race · each model advances independently';
  $('scorecard').innerHTML = data.models.map(m => {
    const s=summaries[m.id];
    return `<tr><td><span class="dot" style="--color:${colors[m.id]}"></span>${m.name}</td><td>${s.n}</td><td>${pct(s.accuracy)}</td><td>${pct(s.confidence)}</td><td>${num(s.latency)}</td><td>${num(s.p95)}</td><td>${s.brier==null?'—':s.brier.toFixed(3)}</td><td>${pct(s.ece)}</td></tr>`;
  }).join('');
  $('qualityHelp').textContent = {
    accuracy:'Share of scored answers that match the dataset label. Higher is better.',
    confidence:'Probability assigned to the predicted answer. High confidence does not guarantee correctness.',
    brier:'Sum of squared probability errors across outcomes. Lower is better.',
    ece:'10-bin calibration error: the gap between confidence and accuracy. Lower is better.'
  }[$('metric').value] + (isRace() ? ' Models currently cover different sample prefixes.' : '');
  draw('quality',$('metric').value); draw('latency','latency');
  if (isRace()) drawThroughput();
}

function canvasContext(id) {
  const canvas=$(id), w=canvas.clientWidth, h=240, ratio=devicePixelRatio||1;
  canvas.width=w*ratio; canvas.height=h*ratio;
  const c=canvas.getContext('2d'); c.scale(ratio,ratio); c.font='10px system-ui';
  return {canvas,c,w,h,left:48,right:w-28,top:15,bottom:h-30};
}

function draw(id,metric) {
  const {canvas,c,w,h,left,right,top,bottom}=canvasContext(id);
  const percent=['accuracy','confidence','ece'].includes(metric);
  const max=percent?1:Math.max(1,...Object.values(history).flat().map(s=>s[metric]||0))*1.1;
  for(let j=0;j<5;j++) {
    const y=top+(bottom-top)*j/4, v=max*(1-j/4);
    c.strokeStyle='#edf0f4'; c.beginPath(); c.moveTo(left,y); c.lineTo(right,y); c.stroke();
    c.fillStyle='#9099a7'; c.fillText(percent?Math.round(v*100)+'%':v.toFixed(metric==='brier'?2:0),4,y+3);
  }
  for(let j=0;j<=4;j++) c.fillText(Math.round(1+(data.total-1)*j/4),left+(right-left)*j/4-5,h-9);
  for(const m of data.models) {
    c.strokeStyle=colors[m.id]; c.lineWidth=2; c.beginPath();
    history[m.id].forEach((s,i)=>{
      if(s[metric]==null)return;
      const x=left+i/(data.total-1||1)*(right-left), y=bottom-s[metric]/max*(bottom-top);
      if(i)c.lineTo(x,y); else c.moveTo(x,y);
      if(i===history[m.id].length-1){c.fillStyle=colors[m.id];c.fillRect(x-2,y-2,4,4);}
    }); c.stroke();
  }
  canvas.onmousemove=e=>{
    const i=Math.min(data.total-1,Math.max(0,Math.round((e.offsetX-left)/(right-left)*(data.total-1))));
    $('tooltip').textContent=`Sample ${i+1}\n`+data.models.map(m=>{
      const v=history[m.id][i]?.[metric];return `${m.name}: ${percent?pct(v):num(v)}`;
    }).join('\n');
    $('tooltip').hidden=false; $('tooltip').style.left=Math.min(innerWidth-170,e.clientX+12)+'px';
    $('tooltip').style.top=Math.max(5,e.clientY-125)+'px';
  };
  canvas.onmouseleave=()=>$('tooltip').hidden=true;
}

function drawThroughput() {
  const {c,h,left,right,top,bottom}=canvasContext('throughput');
  const maxTime=Math.max(1000,elapsed);
  for(let j=0;j<=4;j++) {
    const y=bottom-(bottom-top)*j/4;
    c.strokeStyle='#edf0f4'; c.beginPath(); c.moveTo(left,y); c.lineTo(right,y); c.stroke();
    c.fillStyle='#9099a7'; c.fillText(Math.round(data.total*j/4),4,y+3);
    c.fillText((maxTime/1000*j/4).toFixed(1)+'s',left+(right-left)*j/4-5,h-9);
  }
  for(const m of data.models) {
    c.strokeStyle=colors[m.id];c.lineWidth=2;c.beginPath();c.moveTo(left,bottom);
    for(let i=0;i<counts[m.id];i++) {
      const x=left+timeline[m.id][i]/maxTime*(right-left);
      c.lineTo(x,bottom-i/data.total*(bottom-top));
      c.lineTo(x,bottom-(i+1)/data.total*(bottom-top));
    }
    c.lineTo(left+elapsed/maxTime*(right-left),bottom-counts[m.id]/data.total*(bottom-top));c.stroke();
  }
}

function stop() { playing=false; clearTimeout(timer); $('play').textContent='▶ Play stream'; }
function tick() {
  if (!playing) return;
  const now=performance.now(), delta=now-lastTick;
  lastTick=now; previewClock+=delta;
  if(isLiveInference()) elapsed+=delta;
  else if(isRace()) {
    elapsed=Math.min(elapsed+delta,Math.max(...Object.values(timeline).map(t=>t.at(-1))));
    counts=Object.fromEntries(data.models.map(m=>[m.id,RaceTiming.completedAt(timeline[m.id],elapsed)]));
  }
  const advances=Math.floor(previewClock/Number($('speed').value));
  if(advances){step=Math.min(data.total-1,step+advances);previewClock%=Number($('speed').value);}
  scheduleRender();
  const complete=isRace()?data.models.every(m=>counts[m.id]===data.total):step===data.total-1;
  if(complete){stop();return;}
  timer=setTimeout(tick,Math.min(40,Number($('speed').value)));
}

function configure() {
  // Always in race mode now
  $('modeLabel').textContent='Throughput race';
  $('racePanel').hidden=false;
  $('raceNote').hidden=false;
  $('raceNote').textContent=isLiveInference()
    ? 'Live inference: each model runs the shared queue on this server at its own real speed (the first case includes model loading). Cards, plots and the scorecard use only completed live results.'
    : 'Each model works through the same queue at its own measured speed. The preview rate only controls the question/image viewer. Cards show each model’s latest completed case; their metrics may cover different numbers of questions. Saved-measurement replay, not live inference.';
  $('overviewProgressLabel').textContent='ELAPSED REPLAY TIME';
  $('speed').title='Input preview rate only. Model counters use unscaled measured latency.';
  $('speed').setAttribute('aria-label','Input preview rate');
  [1,5,25].forEach((rate,i)=>{$('speed').options[i].textContent=`${rate} ${dataset==='slake'?'images':'questions'} / sec · preview`;});
  lastInput=null;
}

async function load() {
  stop(); busy=true; $('play').disabled=true;
  const version=++loadVersion, requested=dataset;
  const inferenceMode = $('inferenceMode').value;
  $('runStatus').textContent='Preparing comparison…';
  const response=await fetch(`/api/data?dataset=${requested}&inference=${inferenceMode}`);
  if(!response.ok)throw Error(await response.text());
  const incoming=await response.json();
  if(version!==loadVersion)return;
  await preloadImages(incoming.samples,version);
  if(version!==loadVersion)return;
  data=incoming;step=0;currentImage=null;lastInput=null;previewClock=0;resetRace();
  $('overviewDataset').textContent=dataset==='slake'?'SLAKE':'MedQA';
  $('overviewDescription').textContent=dataset==='slake'?'Visual question answering':'Clinical question answering';
  $('overviewCases').textContent=data.unique.toLocaleString();
  $('overviewModels').textContent=String(data.models.length).padStart(2,'0');
  $('datasetInfo').textContent=`${data.total.toLocaleString()} stream steps · ${data.unique} unique held-out cases`;
  $('taskTag').textContent=dataset==='slake'?'SLAKE / TEST':'MEDQA / TEST';
  $('questionType').textContent=dataset==='slake'?'VISUAL QUESTION ANSWERING':'CLINICAL QUESTION ANSWERING';
  $('pathsTitle').textContent=`${data.models.length} inference paths`;
  $('legend').innerHTML=data.models.map(m=>`<span><i class="dot" style="--color:${colors[m.id]}"></i>${m.name}</span>`).join('');
  $('sources').textContent='Recorded sources: '+Object.values(data.sources).join(' · ');
  busy=false; $('play').disabled=false;configure();await render();
}

$('play').onclick=()=>{
  if(busy)return;
  if(playing){stop();return;}
  if(isRace()?data.models.every(m=>counts[m.id]===data.total):step===data.total-1){step=0;resetRace();}
  playing=true;lastTick=performance.now();previewClock=0;$('play').textContent='Ⅱ Pause';
  if(isLiveInference()){const token=runToken;data.models.forEach(m=>liveWorker(m,token));}
  tick();
};
$('reset').onclick=async ()=>{if(busy)return;stop();step=0;resetRace();await render();};
$('inferenceMode').onchange=()=>{load().catch(showError);};
$('dataset').onchange=()=>{dataset=$('dataset').value;load().catch(showError);};
$('metric').onchange=render; window.onresize=render;
$('seek').oninput=()=>{if(busy)return;stop();step=Number($('seek').value);render();};
async function moveStep(delta){if(busy)return;stop();step=Math.max(0,Math.min(data.total-1,step+delta));await render();}
$('previous').onclick=()=>moveStep(-1); $('next').onclick=()=>moveStep(1);
$('nextImage').onclick=()=>{
  if(busy)return;
  const id=sampleAt(step).image_id;
  for(let i=step+1;i<data.total;i++)if(sampleAt(i).image_id!==id){stop();step=i;render();return;}
};
$('cards').onclick=e=>{
  const button=e.target.closest('[data-model]');
  if(!button)return;
  stop();step=counts[button.dataset.model]-1;render();
};
document.addEventListener('keydown',e=>{
  if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(e.target.tagName))return;
  if(e.key==='ArrowRight'){e.preventDefault();moveStep(1);}
  if(e.key==='ArrowLeft'){e.preventDefault();moveStep(-1);}
});
$('export').onclick=()=>{
  if(busy)return;
  const csv=['model,mode,elapsed_replay_ms,scored,accuracy,confidence,mean_latency_ms,p95_latency_ms,brier,ece',
    ...data.models.map(m=>{const s=summaries[m.id];return [m.name,$('inferenceMode').value,isRace()?elapsed:'',s.n,s.accuracy,s.confidence,s.latency,s.p95,s.brier,s.ece].join(',');})].join('\n');
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  a.download=`sysone-med-${dataset}-${$('inferenceMode').value}-${step+1}.csv`;a.click();URL.revokeObjectURL(a.href);
};
function showError(e){stop();$('runStatus').textContent='Unable to load results: '+e.message;}
load().catch(showError);
