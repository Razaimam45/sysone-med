#!/usr/bin/env python3
"""Download model assets and optional Hugging Face datasets into a local cache."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import zipfile
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data/cache"))
    parser.add_argument("--source-dir", default=str(ROOT / "data/sources/huggingface"))
    parser.add_argument("--normalized-dir", default=str(ROOT / "data/normalized"))
    parser.add_argument("--models-dir", default=str(ROOT / "data/models"))
    parser.add_argument("--manifest", default=str(ROOT / "data/manifests/downloads.json"))
    parser.add_argument("--text-model", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--vision-model", default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--dataset", action="append", default=[], metavar="ID[:CONFIG]",
                        help="Optional HF dataset id and config; repeat for additional datasets")
    parser.add_argument("--revision", help="Optional model revision")
    parser.add_argument("--all-medical", action="store_true",
                        help="Download and normalize MedQA, SLAKE, and PathVQA closed-choice rows")
    parser.add_argument("--pmc-vqa-max-rows", type=int, default=20000,
                        help="Maximum combined PMC-VQA train/test rows to materialize")
    parser.add_argument("--include-pmc-vqa", action="store_true",
                        help="Download PMC-VQA images_2.zip and materialize the capped dataset")
    args = parser.parse_args()
    root = Path(args.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install model dependencies first: python -m pip install -e '.[train]'") from exc
    models = {}
    for label, model_id in (("text", args.text_model), ("vision", args.vision_model)):
        path = snapshot_download(model_id, cache_dir=str(root / "hub"), revision=args.revision)
        alias = Path(args.models_dir) / label
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        alias.symlink_to(os.path.relpath(Path(path).resolve(), alias.parent.resolve()), target_is_directory=True)
        models[label] = {"id": model_id, "path": path, "revision": args.revision}
        print(f"{label}: {model_id} -> {path}", flush=True)
    datasets = {}
    if args.all_medical:
        args.dataset.extend([
            "bigbio/med_qa:med_qa_en_source",
            "openlifescienceai/medmcqa",
            "BoKelvin/SLAKE",
            "flaviagiammarino/path-vqa",
        ])
    if args.dataset:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit("Install training dependencies: python -m pip install -e '.[train]'") from exc
        for spec in args.dataset:
            dataset_id, _, config = spec.partition(":")
            loaded = load_dataset(dataset_id, config or None, cache_dir=str(root / "datasets"))
            destination = root / "datasets" / dataset_id.replace("/", "__")
            loaded.save_to_disk(str(destination))
            datasets[spec] = {"id": dataset_id, "config": config or None, "path": str(destination), "splits": list(loaded)}
            print(f"dataset: {spec} -> {destination}", flush=True)
            if dataset_id in {"BoKelvin/SLAKE", "flaviagiammarino/path-vqa"}:
                from huggingface_hub import snapshot_download
                raw_root = Path(args.source_dir) / dataset_id.replace("/", "__")
                snapshot_download(dataset_id, repo_type="dataset",
                                  local_dir=str(raw_root), revision=args.revision)
                if dataset_id == "BoKelvin/SLAKE":
                    archive = raw_root / "imgs.zip"
                    image_root = raw_root / "imgs"
                    if archive.exists() and not image_root.exists():
                        with zipfile.ZipFile(archive) as bundle:
                            bundle.extractall(raw_root)
        normalized = Path(args.normalized_dir)
        normalized.mkdir(parents=True, exist_ok=True)
        normalize_medical_datasets(args.dataset, root, normalized, Path(args.source_dir))
    if args.include_pmc_vqa and args.pmc_vqa_max_rows > 0:
        download_pmc_vqa(root, Path(args.normalized_dir), args.pmc_vqa_max_rows, Path(args.source_dir))
        datasets["RadGenome/PMC-VQA"] = {
            "id": "RadGenome/PMC-VQA", "config": None,
            "path": str(Path(args.normalized_dir) / "pmc_vqa"),
            "splits": ["train", "test"], "max_rows": args.pmc_vqa_max_rows,
        }
    previous = {}
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
    merged_datasets = dict(previous.get("datasets", {}))
    merged_datasets.update(datasets)
    manifest = {
        "models": models,
        "datasets": merged_datasets,
        "repository_datasets": {
            "pubmedqa": {"path": "data/sources/pubmedqa", "status": "present; loader uses official test IDs"},
            "vqarad": {"path": "data/sources/vqarad", "status": "present; loader filters yes/no and supports image-group splits"},
        },
        "normalized_datasets": {
            directory.name: {"path": str(directory), "splits": {
                path.stem: sum(bool(line.strip()) for line in path.open())
                for path in sorted(directory.glob("*.jsonl"))}}
            for directory in sorted(Path(args.normalized_dir).glob("*"))
            if directory.is_dir()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}", flush=True)


def normalize_medical_datasets(specs: list[str], cache_root: Path, output_root: Path, source_root: Path = ROOT / "data/sources/huggingface") -> None:
    """Write closed-choice rows and detached images in the trainer schema."""
    from datasets import load_from_disk

    for spec in specs:
        dataset_id, _, config = spec.partition(":")
        source_path = cache_root / "datasets" / dataset_id.replace("/", "__")
        dataset = load_from_disk(str(source_path))
        name = dataset_id.split("/")[-1].lower().replace("-", "_")
        out = output_root / name
        (out / "images").mkdir(parents=True, exist_ok=True)
        written = 0
        for split, rows in dataset.items():
            records = []
            for index, row in enumerate(rows):
                options: list[str] = []
                answer = str(row.get("answer", "")).strip()
                state: str | dict = ""
                image_value = row.get("image")
                if dataset_id == "bigbio/med_qa":
                    options = [str(item["value"]) for item in row.get("options", [])]
                    answer_index = next((i for i, item in enumerate(row.get("options", []))
                                         if str(item.get("key")) == str(row.get("answer_idx"))), -1)
                    answer = str(row.get("answer", ""))
                    if answer_index < 0 and answer in options:
                        answer_index = options.index(answer)
                    state = str(row.get("meta_info", ""))
                elif dataset_id == "openlifescienceai/medmcqa":
                    if str(row.get("choice_type", "single")) != "single":
                        continue
                    options = [str(row.get(key, "")) for key in ("opa", "opb", "opc", "opd")]
                    answer_index = int(row.get("cop", -1))
                    # Explanation/subject/topic annotations are not available question inputs.
                    state = ""
                elif dataset_id in {"BoKelvin/SLAKE", "flaviagiammarino/path-vqa"}:
                    if answer.lower() not in {"yes", "no"}:
                        continue
                    options = ["yes", "no"]
                    answer_index = 0 if answer.lower() == "yes" else 1
                    state = ""  # Ground-truth organ/location annotations are not question inputs.
                    image_value = row.get("image") or row.get("img_name")
                else:
                    continue
                if not options or answer_index < 0:
                    continue
                image_path = None
                if image_value is not None:
                    image = image_value
                    if hasattr(image, "save"):
                        image_path = out / "images" / f"{split}_{index}.jpg"
                        existing_png = image_path.with_suffix(".png")
                        if existing_png.exists():
                            image_path = existing_png
                        elif not image_path.exists():
                            image.convert("RGB").save(image_path, format="JPEG", quality=90, optimize=False)
                    elif isinstance(image, str):
                        source = Path(image)
                        if not source.exists():
                            raw_root = source_root / dataset_id.replace("/", "__")
                            direct = raw_root / image
                            nested = raw_root / "imgs" / image
                            if direct.is_file():
                                source = direct
                            elif nested.is_file():
                                source = nested
                            else:
                                matches = [p for p in raw_root.rglob(source.name) if p.is_file()]
                                if len(matches) != 1:
                                    raise ValueError(f"Missing or ambiguous source image: {image}")
                                source = matches[0]
                        if source.exists():
                            relative_image = Path(image)
                            if relative_image.is_absolute() or ".." in relative_image.parts:
                                relative_image = Path(f"{split}_{index}_{source.name}")
                            image_path = out / "images" / relative_image
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source, image_path)
                if dataset_id in {"BoKelvin/SLAKE", "flaviagiammarino/path-vqa"} and image_path is None:
                    raise ValueError(f"Visual example lacks its image: {dataset_id}/{split}/{index}")
                records.append({
                    "id": f"{name}:{split}:{row.get('qid', row.get('id', index))}",
                    "state": {"text": state, "image": str(image_path) if image_path else None},
                    "question": str(row.get("question", "")),
                    "options": options,
                    "label": answer_index,
                    "dataset": name,
                    "source_split": split,
                    "input_schema": "question_options_only_v2" if dataset_id == "openlifescienceai/medmcqa" else "decision_v2",
                })
            target = out / f"{split}.jsonl"
            target.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
            written += len(records)
        print(f"normalized: {spec} -> {out} ({written} closed-choice rows)", flush=True)


def normalize_pmc_choice(value: str, letter: str) -> str:
    text = re.sub(rf'^\s*"?{letter}\s*[:.)-]\s*', '', value).strip()
    # Some CSV rows wrap the entire four-choice block in quotes, leaving a
    # leading quote on A and an unmatched trailing quote on D.
    if text.endswith('"') and text.count('"') % 2:
        text = text[:-1].rstrip()
    return text


def download_pmc_vqa(cache_root: Path, output_root: Path, max_rows: int, source_root: Path = ROOT / "data/sources/huggingface") -> None:
    """Download PMC-VQA's smaller image archive and cap normalized rows."""
    from huggingface_hub import hf_hub_download

    dataset_id = "RadGenome/PMC-VQA"
    raw = source_root / "RadGenome__PMC-VQA"
    raw.mkdir(parents=True, exist_ok=True)
    csv_paths = {
        split: hf_hub_download(dataset_id, f"{split}_2.csv", repo_type="dataset",
                                cache_dir=str(cache_root / "hub"))
        for split in ("train", "test")
    }
    archive = Path(hf_hub_download(dataset_id, "images_2.zip", repo_type="dataset",
                                   cache_dir=str(cache_root / "hub")))
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    if max_rows < 2:
        raise ValueError("PMC-VQA cap must reserve at least one train and one test row")
    test_quota = max(1, max_rows // 10)
    for split in ("train", "test"):
        remaining = test_quota if split == "test" else max_rows - test_quota
        selected = []
        with Path(csv_paths[split]).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if remaining <= 0:
                    break
                answer_label = str(row.get("Answer_label") or row.get("Answer") or "").strip()
                if answer_label not in {"A", "B", "C", "D"}:
                    continue
                selected.append(row)
                remaining -= 1
        rows_by_split[split] = selected
    out = output_root / "pmc_vqa"
    image_out = out / "images"
    image_out.mkdir(parents=True, exist_ok=True)
    wanted = {Path(row["Figure_path"]).name for rows in rows_by_split.values() for row in rows}
    with zipfile.ZipFile(archive) as bundle:
        members = {}
        for name in bundle.namelist():
            basename = Path(name).name
            if basename in wanted and basename in members:
                raise ValueError(f"Ambiguous image basename in PMC archive: {basename}")
            members[basename] = name
        for name in wanted:
            member = members.get(name)
            if member:
                target = image_out / name
                if not target.exists():
                    with bundle.open(member) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
    total = 0
    for split, rows in rows_by_split.items():
        records = []
        for index, row in enumerate(rows):
            options = [normalize_pmc_choice(row[f"Choice {letter}"], letter) for letter in "ABCD"]
            image_file = image_out / Path(row["Figure_path"]).name
            if not image_file.is_file():
                raise FileNotFoundError(f"PMC-VQA archive lacks {image_file.name}")
            records.append({
                "id": f"pmc_vqa:{split}:{index}",
                "state": {"text": "", "image": str(image_out / Path(row["Figure_path"]).name)},
                "question": row["Question"].strip(), "options": options,
                "label": "ABCD".index(str(row.get("Answer_label") or row.get("Answer")).strip()), "dataset": "pmc_vqa",
                "source_split": split,
            })
        (out / f"{split}.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records))
        total += len(records)
    print(f"normalized: {dataset_id} -> {out} ({total} rows, cap {max_rows})", flush=True)


if __name__ == "__main__":
    main()
