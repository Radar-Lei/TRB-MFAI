#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main_v109.tex > build_v109.log 2>&1
bibtex main_v109 >> build_v109.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v109.tex >> build_v109.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v109.tex >> build_v109.log 2>&1
cp main_v109.pdf TRB-MFAI_v109.pdf
cp main_v109.pdf ../TRB-MFAI_v109.pdf
pdfinfo main_v109.pdf | grep Pages
