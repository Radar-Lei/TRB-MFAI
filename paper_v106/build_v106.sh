#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main_v106.tex > build_v106.log 2>&1
bibtex main_v106 >> build_v106.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v106.tex >> build_v106.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v106.tex >> build_v106.log 2>&1
cp main_v106.pdf TRB-MFAI_v106.pdf
cp main_v106.pdf ../TRB-MFAI_v106.pdf
pdfinfo main_v106.pdf
