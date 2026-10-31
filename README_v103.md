# v103 本地复现

使用项目已有 `.venv/bin/python`（NumPy、SciPy、PyTorch、NetworkX）。独立复核另用已有Conda环境cie_msde_2026。LaTeX使用系统pdflatex/bibtex。无需GPU训练、商业求解器或新数据申请。

```bash
.venv/bin/python experiments/run_e113_v103_spatial.py --refine
.venv/bin/python experiments/report_e113_v103.py
.venv/bin/python experiments/run_e114_v103_coverage.py
.venv/bin/python experiments/run_e115_v103_scan.py
bash paper/build_v103.sh
```

Anaheim补算读取原e112_v102/arrays.npz和results.json，复用21个状态；新20个中点逐点checkpoint在e113_v103，已有checkpoint会被复用。原始TNTP输入在experiments/data/Anaheim/。`report_e113_v103.py`从已有数组生成表和中文报告。

覆盖实验协议在e114_v103/protocol.json，全部72个端点和列在arrays.npz；`run_e114_v103_coverage.py --aggregate`只从端点重聚合几何统计，不重求均衡。几何扫描使用e87_v85_newod_refine.npz的六列，120列计时为重复六列20次，不能当新增科学样本。

独立审查：review-stage/v103/independent_code_check.py与recheck_paper_numbers.py，只读复核原始数据，不重跑均衡。全量学习沿用v102，见README_v102.md；本轮没有新训练。

最终PDF为项目根目录TRB-MFAI_v103.pdf；主源码paper/main_v103.tex，旧版保持。v103所有理论、设置与补充数值随同一PDF附录，不拆独立Supplementary。
