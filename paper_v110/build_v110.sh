#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main_v110.tex > build_v110.log 2>&1
bibtex main_v110 >> build_v110.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v110.tex >> build_v110.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v110.tex >> build_v110.log 2>&1
cp main_v110.pdf TRB-MFAI_v110.pdf
cp main_v110.pdf ../TRB-MFAI_v110.pdf
pdfinfo main_v110.pdf | grep Pages
