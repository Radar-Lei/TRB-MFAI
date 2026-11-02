#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main_v111.tex > build_v111.log 2>&1
bibtex main_v111 >> build_v111.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v111.tex >> build_v111.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v111.tex >> build_v111.log 2>&1
cp main_v111.pdf TRB-MFAI_v111.pdf
cp main_v111.pdf ../TRB-MFAI_v111.pdf
pdfinfo main_v111.pdf | grep Pages
