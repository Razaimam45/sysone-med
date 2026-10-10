# SysOne-Med · demo2

A small, dependency-free website that presents the **SysOne study** and lets you
*try* the five decision variants in two settings.

```
demo2/
├── server.py            # static + JSON server (no GPU, no API key, no build)
└── static/
    ├── index.html       # home page: study briefing + paper figures
    ├── home.js          # fills the briefing with real per-variant numbers
    ├── style.css        # shared look (teal/ink, matches demo/)
    ├── snake.html       # the snake race
    ├── snake.css
    └── snake.js         # the game: each snake moves at its measured decision speed
```

## Run

```bash
python demo2/server.py            # → http://localhost:8766
python demo2/server.py --port 9000
```

Then open:

| Route | What it is |
| --- | --- |
| `/` | **Home** — a briefing on the SysOne study: the five decision paths, real per-variant latency/accuracy, and the paper's figures. |
| `/vqa` | **Medical VQA / QA** — the full inference observatory, reused from `demo/` (SLAKE + MedQA, aligned comparison + throughput race). |
| `/snake` | **Snake race** — five snakes in one arena; each snake makes a move per decision and a decision costs that variant's *measured mean latency*, so the fastest path visibly outruns the slowest. |

## What it shows

The home page and the snake race are driven by the **same real measurements** in
`demo/data/comparison.json` (the recorded predictions the original demo uses).
Per variant we report:

- **mean decision latency** (ms) — the number that sets each snake's speed,
- **held-out accuracy** — across the recorded SLAKE + MedQA cases,
- a **speed rank** (fastest → slowest).

The snake race rescales on-screen speed into a readable range (the fastest snake
is not literally 300× the slowest on screen), but the leaderboard shows the real
`ms/decision` and `moves/s` for every variant.

## The five variants

| id | name | decision mechanism |
| --- | --- | --- |
| `medjev` | MedJev | direct decision · LoRA (text-only) |
| `gen` | AR Gen | autoregressive · 32-token generation |
| `ll` | AR LL | autoregressive · single-token generation |
| `head` | AR Head | frozen MedGemma backbone + learned head |
| `nar` | NAR | non-autoregressive scoring |

## Requirements

- Python 3 (standard library only — no `pip install`).
- `demo/data/comparison.json` must exist. If it doesn't, run the original demo
  once to build it: `python demo/server.py` (it writes the file on startup).
- Paper figures are read from `demo2/figures/` (PNGs converted from the paper PDFs).

## GitHub Pages (static site)

GitHub Pages can't run the Python server, so `build_pages.py` produces a fully
static, self-contained copy of the site in `demo2/site/`:

- precomputes the API responses into `data/snake.json`, `data/slake.json`,
  `data/med_qa.json`,
- copies the paper figures and the 96 unique SLAKE scans,
- rewrites every page to use **relative** paths (so it works at any sub-path).

```bash
python demo2/build_pages.py        # → demo2/site/  (~14 MB)
```

To publish, push `demo2/site/` to a `gh-pages` branch (see the repo README for
the exact commands). The VQA "live inference" mode is omitted on Pages (no
GPU/server) — stored replay only.

## Notes

- Everything replays **saved measurements**; no live inference, GPU, or API key.
- The server binds to `127.0.0.1`.
- The VQA app is served straight from `demo/static/`, so it stays in sync with the
  original demo automatically.
