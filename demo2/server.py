#!/usr/bin/env python3
"""SysOne-Med · demo2 — study briefing + two applications (medical VQA + snake race).

A dependency-free static/JSON server that:
  * serves the home page (a briefing on the SysOne study with paper figures),
  * reuses the existing demo/ VQA+QA observatory under /vqa,
  * serves a side-by-side "snake race" that pits every decision variant against
    each other at its *measured* per-decision latency,
  * serves the paper figures (converted from the paper PDFs) and the recorded
    comparison data.

No GPU, API key, or web build is required. Everything replays saved measurements.

Run:
    python demo2/server.py            # http://localhost:8766
    python demo2/server.py --port 9000
"""
import argparse
import json
import mimetypes
import statistics
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
DEMO_STATIC = ROOT / 'demo' / 'static'
FIGURES = Path(__file__).resolve().parent / 'figures'
COMPARISON = ROOT / 'demo' / 'data' / 'comparison.json'
STATIC = Path(__file__).resolve().parent / 'static'

# Human-readable identity for each decision variant, shared by the briefing and
# the snake race. `id` keys match the recorded ledgers in comparison.json.
VARIANTS = [
    dict(id='medjev', name='MedJev',     tag='Direct decision · LoRA',   color='#a17bd3',
         blurb='A compact causal LM that scores the candidate set directly — no answer tokens are written.'),
    dict(id='gen',    name='AR Gen',     tag='Autoregressive · 32 tokens', color='#d69957',
         blurb='MedGemma writes free text up to 32 tokens, one at a time, then the text is parsed into a choice.'),
    dict(id='ll',     name='AR LL',      tag='Autoregressive · 1 token', color='#4fc3c9',
         blurb='The same MedGemma backbone emits exactly one token — the fastest of the autoregressive paths.'),
    dict(id='head',   name='AR Head',    tag='Frozen backbone + head',   color='#658ed0',
         blurb='A frozen MedGemma backbone feeds a small trained pointer head that scores every option at once.'),
    dict(id='nar',    name='NAR',        tag='Non-autoregressive',       color='#48a895',
         blurb='A non-autoregressive model encodes text + image and a shared scorer scores all candidates in a single forward pass.'),
]
VARIANT_BY_ID = {v['id']: v for v in VARIANTS}


def _load_comparison():
    if not COMPARISON.exists():
        raise SystemExit(
            f'Missing {COMPARISON}. Run the original demo once to build it:\n'
            f'    python demo/server.py   (it writes demo/data/comparison.json on startup)')
    return json.loads(COMPARISON.read_text())


def _variant_stats(bundle):
    """Per-variant mean latency + accuracy across both recorded datasets.

    These measured numbers are what drive the snake race: each variant's snake
    moves once per decision, and a decision costs its real mean latency.
    """
    stats = {}
    for v in VARIANTS:
        lats, correct, n = [], 0, 0
        for ds in ('slake', 'med_qa'):
            recs = bundle.get(ds, {}).get('recorded', {}).get(v['id'], {})
            for r in recs.values():
                if r.get('latency_ms') is not None:
                    lats.append(r['latency_ms'])
                n += 1
                correct += 1 if r.get('correct') else 0
        if not lats:
            continue
        mean = statistics.mean(lats)
        stats[v['id']] = dict(
            mean_latency_ms=round(mean, 1),
            p50_latency_ms=round(statistics.median(lats), 1),
            accuracy=round(correct / n * 100, 1) if n else None,
            decisions=len(lats),
            # Snake moves per second = 1000 / mean decision latency.
            moves_per_sec=round(1000.0 / mean, 3),
        )
    return stats


def build_snake_config(bundle):
    stats = _variant_stats(bundle)
    # Only race variants that actually have recorded decisions.
    active = [v for v in VARIANTS if v['id'] in stats]
    # A shared, human-readable tick scale: the slowest variant sets the base so
    # every snake is visible, but we compress the range so the fastest is not
    # absurdly fast. We map measured moves/sec into a bounded on-screen range.
    mps = [stats[v['id']]['moves_per_sec'] for v in active]
    lo, hi = min(mps), max(mps)
    # Map [lo, hi] -> [MIN_MPS, MAX_MPS] on-screen moves per second.
    MIN_MPS, MAX_MPS = 1.2, 14.0
    def scale(x):
        if hi == lo:
            return (MIN_MPS + MAX_MPS) / 2
        return MIN_MPS + (x - lo) / (hi - lo) * (MAX_MPS - MIN_MPS)
    variants = []
    for v in active:
        s = stats[v['id']]
        variants.append(dict(
            id=v['id'], name=v['name'], tag=v['tag'], color=v['color'], blurb=v['blurb'],
            mean_latency_ms=s['mean_latency_ms'], p50_latency_ms=s['p50_latency_ms'],
            accuracy=s['accuracy'], decisions=s['decisions'],
            measured_moves_per_sec=s['moves_per_sec'],
            on_screen_moves_per_sec=round(scale(s['moves_per_sec']), 3),
        ))
    # Rank by measured speed (fastest first) for the briefing scoreboard.
    variants.sort(key=lambda x: x['mean_latency_ms'])
    return dict(
        variants=variants,
        grid=dict(cols=17, rows=11),
        note=('Each snake makes one move per decision. A decision costs the variant\'s '
              'measured mean latency, so faster variants outrun slower ones. On-screen '
              'speed is rescaled into a readable range; the measured ms values are the real numbers.'),
    )


class Handler(BaseHTTPRequestHandler):
    def _send(self, data, mime, cache='no-store'):
        if isinstance(data, str):
            data = data.encode()
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Cache-Control', cache)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, path, cache='no-store'):
        if not path.is_file():
            self.send_error(404); return
        self._send(path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'application/octet-stream', cache)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            # ---- API ----
            if path == '/api/snake':
                self._send(json.dumps(self.server.snake, allow_nan=False), 'application/json')
            elif path == '/api/data':
                dataset = parse_qs(parsed.query).get('dataset', ['slake'])[0]
                if dataset not in self.server.bundle:
                    self.send_error(404, f'unknown dataset {dataset}'); return
                self._send(json.dumps(self.server.bundle[dataset], allow_nan=False), 'application/json')
            elif path == '/image':
                # Reuse the demo's image index (SLAKE scans).
                ident = parse_qs(parsed.query).get('id', [''])[0]
                p = self.server.image_paths.get(ident)
                if not p:
                    self.send_error(404); return
                self._send(p.read_bytes(), mimetypes.guess_type(p.name)[0] or 'image/jpeg', 'private, max-age=3600')
            # ---- Paper figures ----
            elif path.startswith('/figures/'):
                name = Path(path).name  # strip any traversal
                self._serve_file(FIGURES / name, 'private, max-age=86400')
            # ---- VQA app (reused from demo/static) ----
            # The VQA page uses relative asset paths (style.css, race.js, app.js),
            # so it must be served at /vqa/ (trailing slash) for them to resolve
            # under /vqa/. Redirect the slash-less form.
            elif path == '/vqa':
                self.send_response(302)
                self.send_header('Location', '/vqa/')
                self.send_header('Content-Length', '0')
                self.end_headers()
            elif path == '/vqa/':
                self._serve_file(DEMO_STATIC / 'index.html')
            elif path.startswith('/vqa/'):
                name = Path(path[len('/vqa/'):]).name
                self._serve_file(DEMO_STATIC / name)
            # ---- demo2 static (home + snake) ----
            elif path in ('/', '/index.html'):
                self._serve_file(STATIC / 'index.html')
            elif path in ('/snake', '/snake/'):
                self._serve_file(STATIC / 'snake.html')
            else:
                name = Path(path.lstrip('/')).name
                self._serve_file(STATIC / name)
        except Exception as exc:  # noqa: BLE001
            self.send_error(500, str(exc).replace('\n', ' ')[:300])

    def log_message(self, *_):
        pass


def _image_index():
    """Map SLAKE image ids to absolute paths, mirroring demo/server.py."""
    index = {}
    prepared = ROOT / 'data' / 'prepared' / 'test.jsonl'
    if not prepared.exists():
        return index
    for line in prepared.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get('dataset') == 'slake' and r.get('state', {}).get('image'):
            index[r['id']] = Path(r['state']['image']).resolve()
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()

    bundle = _load_comparison()
    snake = build_snake_config(bundle)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.bundle = bundle
    server.snake = snake
    server.image_paths = _image_index()

    print('SysOne-Med · demo2 → http://localhost:%d' % args.port, flush=True)
    print('  home (briefing)  → /', flush=True)
    print('  medical VQA/QA   → /vqa', flush=True)
    print('  snake race       → /snake', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == '__main__':
    main()
