#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main_v112.tex > build_v112.log 2>&1
bibtex main_v112 >> build_v112.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v112.tex >> build_v112.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v112.tex >> build_v112.log 2>&1
cp main_v112.pdf TRB-MFAI_v112.pdf
cp main_v112.pdf ../TRB-MFAI_v112.pdf
pdfinfo main_v112.pdf | grep Pages
