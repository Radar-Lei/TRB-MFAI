#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main_v108.tex > build_v108.log 2>&1
bibtex main_v108 >> build_v108.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v108.tex >> build_v108.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_v108.tex >> build_v108.log 2>&1
cp main_v108.pdf TRB-MFAI_v108.pdf
cp main_v108.pdf ../TRB-MFAI_v108.pdf
pdfinfo main_v108.pdf | grep Pages
