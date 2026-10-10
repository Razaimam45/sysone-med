#!/usr/bin/env python3
"""Build a static, GitHub-Pages-ready copy of the SysOne-Med demo2 site.

GitHub Pages cannot run the Python server, so this precomputes the API
responses into static JSON files and rewrites every page to use *relative*
paths. The output in ``demo2/site/`` is fully self-contained and deployable
as-is (just push it to a ``gh-pages`` branch).

Layout produced:
    site/
      index.html  style.css  home.js          # briefing
      snake.html  snake.css  snake.js         # snake race
      figures/*.png                              # paper figures
      data/snake.json  data/slake.json  data/med_qa.json
      vqa/index.html  style.css  app.js  race.js  images/*.jpg

Run:
    python demo2/build_pages.py
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # demo2/
ROOT = HERE.parent                              # project root
sys.path.insert(0, str(HERE))
import server  # noqa: E402  (reuses build_snake_config + _load_comparison)

SITE = HERE / 'site'
PREPARED = ROOT / 'data' / 'prepared' / 'test.jsonl'
FIG_SRC = HERE / 'figures'
DEMO_STATIC = ROOT / 'demo' / 'static'
D2_STATIC = HERE / 'static'


def read(p: Path) -> str:
    return p.read_text()


def write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _slake_image_index():
    """Map SLAKE sample id -> (folder_name, image_path), like demo/server.py."""
    info = {}
    for line in PREPARED.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get('dataset') != 'slake':
            continue
        ip = r.get('state', {}).get('image')
        if ip:
            info[r['id']] = (Path(ip).parent.name, Path(ip))
    return info


def build() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / 'data').mkdir(parents=True)
    (SITE / 'vqa' / 'images').mkdir(parents=True)

    # ---- 1. Precompute the API responses as static JSON ----
    bundle = server._load_comparison()
    snake = server.build_snake_config(bundle)
    write(SITE / 'data' / 'snake.json', json.dumps(snake, allow_nan=False))

    img_info = _slake_image_index()
    for ds in ('slake', 'med_qa'):
        data = bundle[ds]
        if ds == 'slake':
            for s in data['samples']:
                s['image_url'] = ('images/' + img_info[s['id']][0] + '.jpg') \
                    if s['id'] in img_info else None
        write(SITE / 'data' / f'{ds}.json', json.dumps(data, allow_nan=False))

    # ---- 2. Copy the unique SLAKE scans (named by folder) ----
    seen = set()
    for name, path in img_info.values():
        if name in seen:
            continue
        seen.add(name)
        copy(path, SITE / 'vqa' / 'images' / f'{name}.jpg')

    # ---- 3. Copy the paper figures ----
    for f in sorted(FIG_SRC.glob('*.png')):
        copy(f, SITE / 'figures' / f.name)

    # ---- 4. Home / briefing page ----
    copy(D2_STATIC / 'style.css', SITE / 'style.css')
    home = read(D2_STATIC / 'index.html')
    home = home.replace('href="/vqa/"', 'href="vqa/"')
    home = home.replace('src="/figures/', 'src="figures/')
    home = home.replace('href="/snake"', 'href="snake.html"')
    home = home.replace('href="/" class="brand"', 'href="index.html" class="brand"')
    write(SITE / 'index.html', home)

    hj = read(D2_STATIC / 'home.js')
    hj = hj.replace("fetch('/api/snake')", "fetch('data/snake.json')")
    hj = hj.replace("fetch('/api/data?dataset=slake')", "fetch('data/slake.json')")
    write(SITE / 'home.js', hj)

    # ---- 5. Snake race page ----
    copy(D2_STATIC / 'snake.css', SITE / 'snake.css')
    snake_html = read(D2_STATIC / 'snake.html')
    snake_html = snake_html.replace('href="/vqa/"', 'href="vqa/"')
    snake_html = snake_html.replace('href="/snake"', 'href="snake.html"')
    snake_html = snake_html.replace('href="/" class="brand"', 'href="index.html" class="brand"')
    write(SITE / 'snake.html', snake_html)

    sj = read(D2_STATIC / 'snake.js')
    sj = sj.replace("fetch('/api/snake')", "fetch('data/snake.json')")
    write(SITE / 'snake.js', sj)

    # ---- 6. Medical VQA page (reused from demo/static) ----
    copy(DEMO_STATIC / 'style.css', SITE / 'vqa' / 'style.css')
    copy(DEMO_STATIC / 'race.js', SITE / 'vqa' / 'race.js')
    vqa_html = read(DEMO_STATIC / 'index.html')
    vqa_html = vqa_html.replace('href="/" class="brand"', 'href="../index.html" class="brand"')
    # Pages has no GPU/server: drop the live-inference option (stored replay only).
    vqa_html = vqa_html.replace(
        '<option value="stored">Stored inference</option><option value="live">Run inference</option>',
        '<option value="stored">Stored inference</option>')
    write(SITE / 'vqa' / 'index.html', vqa_html)

    aj = read(DEMO_STATIC / 'app.js')
    aj = aj.replace(
        "fetch(`/api/data?dataset=${requested}&inference=${inferenceMode}`)",
        "fetch(`../data/${requested}.json`)")
    write(SITE / 'vqa' / 'app.js', aj)

    # ---- summary ----
    n_img = len(list((SITE / 'vqa' / 'images').glob('*.jpg')))
    n_fig = len(list((SITE / 'figures').glob('*.png')))
    total = sum(f.stat().st_size for f in SITE.rglob('*') if f.is_file())
    print(f'Built {SITE}')
    print(f'  paper figures : {n_fig}')
    print(f'  slake scans   : {n_img}')
    print(f'  total size    : {total / 1e6:.2f} MB')


if __name__ == '__main__':
    build()
