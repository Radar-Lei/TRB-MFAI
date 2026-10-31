#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main_v107.tex > build_v107.log 2>&1
bibtex main_v107 >> build_v107.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v107.tex >> build_v107.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v107.tex >> build_v107.log 2>&1
cp main_v107.pdf TRB-MFAI_v107.pdf
cp main_v107.pdf ../TRB-MFAI_v107.pdf
pdfinfo main_v107.pdf
