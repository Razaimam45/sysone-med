# SysOne-Med · biomedia

From the project root:

```bash
python demo/server.py
```

Open **http://localhost:8765**. Restart an older server and refresh the browser
when updating the demo. No collection command, API key, GPU, or web build is
needed for either playback mode.

## Two working modes

- **Aligned comparison:** all model results appear together for each question.
- **Throughput race:** models independently process the same queue using their
  saved per-case latencies. Each has its own counter, progress bar, and latest
  result. A shared elapsed-time plot shows completed samples. Pause/resume
  preserves the race; Reset starts every model at zero.

The rate selector shows 1, 5, or 25 images per second (questions for MedQA).
In the race this controls only the input preview, not model throughput.
Use **Inspect latest** on a model card to view that model's most recent case.
Quality metrics in the race can cover different sample counts. Both modes
replay saved measurements; neither performs fresh inference.

All unique VQA scans are preloaded and decoded before Play becomes available.
Repeated questions reuse the same image URL and decoded image element. The
server indexes image paths once and allows browser caching, so fast playback
does not issue a new image request for every frame.

Both modes use the same complete saved predictions, including the collected
MedJev results. Accuracy, confidence, calibration, and latency are never invented.
The server validates and writes `demo/data/comparison.json` on startup, then serves
that snapshot consistently throughout the session.

## Included coverage

| Task | Models | Unique questions | Stream steps |
| --- | --- | ---: | ---: |
| MedQA | MedJev, AR Gen, AR Head, AR LL, NAR (Laya-Med) | 1,000 | 1,000 |
| SLAKE | AR Gen, AR Head, AR LL, NAR (Laya-Med) | 355 | 1,000 |

SLAKE cycles through the 355 held-out cases, interleaving image groups. Each
question retains its actual image, choices, label, and predictions. Repeats are
labeled and are not independent measurements. Use arrows, keyboard arrow keys,
the timeline, or **Next image** to navigate.

MedJev's completed 1,000 MedQA measurements are imported from
`demo/runs/live.jsonl`, matched by sample ID and model key. Old hosted Jev records
are excluded. Other sources are the existing AR Gen, AR Head, AR LL, and NAR
prediction ledgers listed in the page's methodology panel. Validated collected
records take precedence over older records for the same model and sample. Missing
predictions or missing latency stop preparation with an explicit error instead of
silently stalling one model. Original source files are preserved.

Confidence is the predicted answer's probability. AR generation keeps invalid
output mass and scores correctness using the generated prediction. Saved logits
use their saved temperature; collected probabilities are preserved. Brier sums
squared probability error; ECE uses 10 equal-width confidence bins; P95 uses
nearest rank. Original hardware and batching conditions can differ between saved
runs. Replay speed does not change reported latency. CSV exports contain the
currently displayed cumulative metrics.

Optional: `--port 9000` changes the port, and `--samples 100` selects a shorter
stream. The server binds to localhost. `collect.py` remains an optional developer
utility for new measurements; neither UI mode requires running it.

Validation:

```bash
python -m unittest discover -s demo -p 'test_*.py'
node --check demo/static/app.js
node demo/test_race.cjs
```
