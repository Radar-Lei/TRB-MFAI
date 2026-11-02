# v109 配套复现包更新

v109 为 v108 外部评审（86/100 综合）后的全项大修版。主线不变：**刻画何时可共享边界值 → 构造误差受控的最简空间剖面 → 说明并改善学习模型生成这些解释的准确性**；本轮按评审要求补齐学习部分的公平比较与收敛证据：六监督目标公平 λ 网格 Pareto（E134）、训练预算 3× 收敛验证（E135）、Anaheim 配对 bootstrap（E136）、response-label 预算曲线（E137）、跨网络归一化总误差表（E138）。论文改名 "Topology-aware, error-controlled spatial explanations of OD demand effects in traffic assignment"，摘要/贡献段/Section 6（正面交付优先）按评审给定版本重排，实验设定改描述性命名（no-background / frozen-background / responsive-background），Table 5/8/12 移附录、Tables 13+14 合并为跨网络汇总。

## v109 新增对象映射

| v109 对象 | 生成脚本 | 冻结输入/输出 |
|---|---|---|
| tab:fairmethods 六方法公平比较 + fig:v109pareto（Pareto 前沿）+ λ* 选择协议 | scripts/run_e134_v109_pareto.py + e134_137_jobs.sh | 六目标（flow/response/wresponse/kboundary/edge/cut）×λ{0.001,0.003,0.01,0.03,0.1,0.3,1}×5 种子=175 续训（e111 预训练检查点、1200 步、lr 3e-5、batch 32+8、每 100 步验证）；统一 v104 选择规则（候选池含 step 0，flow ≤1.1×initial 筛下 min val K-MSE）；λ* 仅验证选择（seed-median flow 约束下 min seed-median val K-MSE）；复现门：response λ=1 与 flow 的 96 路径中位 vs e119 abs_dev=0.0、flow 选择步=v104 冻结。**结果：response λ*=1 综合最优（col 14.19/geo 5.75/bnd 12.08/flow 0.0292）；weighted response λ*=0.003 与之打平（col 13.85/bnd 11.92 为所选权重中最低，各指标差 <0.5pp），λ>0.03 过 flow 筛回退初始；edge/cut λ*=0.3 达 16.23/16.26；boundary seminorm λ*=0.3 验证边界误差全场最低但 test col 24.18 劣于 flow-only（8/96）**。results/e134_v109_pareto/{selection.json（全 λ 候选与逐步选择）, results.json（final+pareto+逐种子改善+门）, kwscale.npy}；175 个训练曲线 cost.json 与检查点不入包（脚本+种子可再生，中位 246–497 s/次） |
| fig:v109convergence 收敛曲线 + Section 7 稳定性段 + C.13 扩展预算协议 | scripts/run_e135_v109_extbudget.py + e135_jobs.sh | 冻结续训协议预算 3× 至 3600 步（SF+Anaheim 各 flow/response×5 种子）；SF 前 1200 步验证轨迹与 e111 逐位一致（断言 <1e-12）；eval 96/48 测试路径。**SF col 21.18→19.46（flow）vs 14.19→11.41（response）；Anaheim 1.64→1.24 vs 1.31→0.88；双网络 response 优势保持且比值 0.67→0.59 / 0.80→0.71**。results/e135_v109_extbudget/{an,sf}_{flow,response}_{s}_cost.json（验证曲线）+ {an,sf}_summary.json + rows |
| Anaheim bootstrap 不确定性句（Section 7 Anaheim 段） | scripts/run_e136_v109_anaheim_bootstrap.py | e131 rows.json 路径级配对 + 10k bootstrap（seed 20261031）：col −0.288 CI[−0.329,−0.213] 48/48、geo −0.172 CI[−0.194,−0.105] 47/48、5/5 种子同向；results/e136_v109_anaheim_bootstrap/results.json |
| fig:v109labelbudget 标签预算曲线 + Section 7 预算段 | scripts/run_e137_v109_labelbudget.py（collect 的 q64 点复用 E134 response λ=1 eval） | per-query {6,12,24,48} 前缀子集（36–288 中心）+64/query=384 复用 E134；λ=E134 验证选择 1.0；**col 17.82/16.64/15.61/14.59/14.19：36 中心已获 ≈48% 改善、288 中心 94%**；results/e137_v109_labelbudget/（results.json curve + q*_cost.json + eval_q*.json） |
| tab:totalerror 跨网络合并表（Anaheim 块） | scripts/run_e138_v109_crossnetwork.py | e130/e131 冻结检查点与 e129 参考列重算（逐行门= e131 至 1e-9）；补 fam_scale_reference（=max\|aS\|，即 geometry_pct 分母）、E_single/E_equal（镜像 e127 定义）；统一归一：逐行 E/fam_scale → path 内平均 → 48 路径中位 ×100；**flow 0.94/4.33/4.57/48.0/46.6/9.9、response 0.78/4.32/4.55/48.0/46.6/9.9**；results/e138_v109_crossnetwork/{results.json, rows.json} |
| 图 fig_pareto/fig_convergence/fig_labelbudget（v109 三新图） | scripts/fig_{pareto,convergence,labelbudget}_v109.py | 分别读 e134 results.json / e135 *_cost.json / e137 results.json；figures/fig_v109_*.pdf |
| 图 fig2（空间分辨率视角双面板）/ fig3（screenline 减负）/ fig_resp_field（主文 2 panel + 附录 active-link 单图） | scripts/fig2_frontier_v109.py / fig3_screenlines_v109.py / fig_resp_field_v109.py | 冻结数组零新求解；宽度尺度与色标跨 panel 统一 |
| 论文全文 | paper_v109/（main_v109.tex 及 \input 链 26 文件、references_v109.bib、build_v109.sh、highlights_v109.tex、cas-sc 三件套、TRB-MFAI_v109.pdf） | — |

## 关键协议要点

- **E134 公平性**：六目标共享初始化/步数/batch/验证节奏/选择规则；λ 网格一致；test 只做最终报告。λ* 只用验证数据。加权响应/边界半范数目标以模型自身流态的 BPR 边际权重构成 D^L J^L − D^S J^S（TSTTT 梯度系数 t0·(1+0.75(F/cap)⁴)，Anaheim 同）；六目标在合适权重下均可训练，weighted response 容许权重区间较窄（λ>0.03 越过 flow 筛）。
- **E137 预算口径**：训练中心是冻结 e111 per-query 顺序的前缀；每中心成本 = 1 次参考流求解 + 1 次灵敏度求解；96 个验证中心为全方法共享固定成本，不计入逐预算成本。
- **E138 归一化**：与 e127（SF）完全同管线（逐行归一→path 均值→路径中位×100）；Anaheim E_single/E_equal 为学习剖面上的全族中点单值与同段数等距分段（中点代表），误差对 aS 计。
- **e135 复现门**：SF 前 12 个验证点与 e111 冻结轨迹一致（<1e-12），扩展是冻结协议的精确延续； Anaheim 选择规则沿用 e130（无 step-0 候选、fallback min-flow）。

运行布局与历史版本一致：scripts/ 置于 experiments/，results/ 置于 experiments/results/，网络输入在 data/。E134 的 select() 需读取冻结选择记录 results/v104_learning/results.json（632 KB，已入包；生成脚本 run_v104_learning.py / report_v104_learning.py 亦入包）。绘图脚本按"含 experiments/results 的祖先目录"自动解析仓库根（figures/src/ 与 experiments/ 两种布局均可），图输出写至 figures/。编译入口 bash build_v109.sh（在 paper_v109/ 内）。E134/E137 为完整再生路径（175+20 次续训，多进程小时级）；E135 为 20 次 3600 步续训（Anaheim 30–105 min/次、SF 4–6 min/次单线程）；E136/E138 为冻结数组再计算（分钟级）。本包为本地同步版本；公开版本化归档（GitHub + DOI）以上传为准，论文可复现性声明以实际归档状态为准。

## round-1 审稿修正同步（2026-10-31 夜）

- **加权目标权重修正**（BLOCKER）：run_e134_v109_pareto.py 的 wL/wS 与 scales()/kwscale.npy 由 F·t(F)（0.15 系数）改为 TSTT 边际 t0·(1+0.75(F/c)⁴)（train 与尺度一致）；70 个 wresponse/kboundary 重训 + eval + select + collect。修复后两目标均可训练（wresponse λ*=0.003 与 response 打平、kboundary λ*=0.3 test 无净优势），tab:fairmethods 与 Pareto 图按新数据更新。
- **E137 协议对齐**：抽样改与 E134 逐字节同规则（有放回、批量 8）；响应尺度 js 改从各预算自身前缀中心计算（不再读全 384 中心的 jscale）；20 重训 + eval + collect。
- **发布包补齐**：v104_learning/results.json + 生成脚本入包；6 个图脚本根目录解析改祖先搜索。
- 论文：learning_protocol_v109 的 D 定义改 diag(∇T)（含 TSTT 显式形式）；收敛段前缀复现声明限定为"前 12 个验证点的验证边界误差 <1e-12"。
