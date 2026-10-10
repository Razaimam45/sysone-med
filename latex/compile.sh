#!/usr/bin/env bash
# Compile paper_results.tex from the project root and place the PDF in latex/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
rm -f paper_results.aux paper_results.bbl paper_results.blg paper_results.log paper_results.out
/home/razaimam/miniconda3/envs/ttw/bin/tectonic paper_results.tex
cp -f paper_results.pdf "$ROOT/latex/paper_results.pdf"
echo "PDF: $ROOT/latex/paper_results.pdf"