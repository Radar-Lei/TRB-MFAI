#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main_v103.tex > build_v103.log 2>&1
bibtex main_v103 >> build_v103.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v103.tex >> build_v103.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v103.tex >> build_v103.log 2>&1
cp main_v103.pdf TRB-MFAI_v103.pdf
cp main_v103.pdf ../TRB-MFAI_v103.pdf
pdfinfo main_v103.pdf
