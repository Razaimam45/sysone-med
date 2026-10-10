import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_trial import persist_results


def test_persist_results_routes_rows_to_baseline_subdirs(tmp_path):
    """Each (model, mode) pair lands in results/<baseline>/results.json."""
    persist_results([{"model": "google/medgemma-4b-it", "task": "pubmedqa",
                      "mode": "conventional_llm"}],
                    tmp_path, "2026-09-25T12:00:00+00:00")
    persist_results([{"model": "convaiinnovations/laya", "task": "med_qa",
                      "mode": "non_ar_decision"}],
                    tmp_path, "2026-09-25T13:00:00+00:00")

    ar_rows = json.loads((tmp_path / "medgemma_ar" / "results.json").read_text())
    laya_rows = json.loads((tmp_path / "laya" / "results.json").read_text())
    assert [row["model"] for row in ar_rows] == ["google/medgemma-4b-it"]
    assert ar_rows[0]["timestamp"] == "2026-09-25T12:00:00+00:00"
    assert [row["model"] for row in laya_rows] == ["convaiinnovations/laya"]
    assert laya_rows[0]["timestamp"] == "2026-09-25T13:00:00+00:00"

    with (tmp_path / "medgemma_ar" / "results.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["timestamp"] == "2026-09-25T12:00:00+00:00"
    assert "Timestamp (UTC)" in (tmp_path / "medgemma_ar" / "report.md").read_text()


def test_persist_results_preserves_legacy_rows_in_same_baseline(tmp_path):
    """Rows without a timestamp (legacy) are kept when a new run appends."""
    legacy = {"model": "google/medgemma-4b-it", "task": "pubmedqa", "mode": "conventional_llm"}
    (tmp_path / "medgemma_ar" / "results.json").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "medgemma_ar" / "results.json").write_text(json.dumps([legacy]))

    persist_results([{"model": "google/medgemma-4b-it", "task": "med_qa",
                      "mode": "conventional_llm"}],
                    tmp_path, "2026-09-25T12:00:00+00:00")

    rows = json.loads((tmp_path / "medgemma_ar" / "results.json").read_text())
    assert [row["model"] for row in rows] == ["google/medgemma-4b-it", "google/medgemma-4b-it"]
    assert rows[0].get("timestamp") is None
    assert rows[1]["timestamp"] == "2026-09-25T12:00:00+00:00"