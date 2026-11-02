# v112 配套复现包更新

v112 为 v111 外部评审（8.4/10，"科学内核已达可投稿水平；主要问题是贡献太多、信息密度太高"）后的贡献聚焦版。论文收拢为**一个总命题、三层贡献**（principal structural / operational / learning），AI 主线贯通（"What the profile measures" 解释框 + 三角色句 s∈U 证书/验证/训练 + fig:pipeline 四面板总流程图替代原 Table 1/2），新增 E150–E153 四计算（两个真实验 + 两个纯重汇总），P0×7 + P1 全部落地。codex nightmare 三轮外审 8.5→8.8→9.1/10（almost，停止条件闭环）。

**新实验与对象映射**

| 实验 | 脚本（experiments/） | 结果 | 论文位置 |
|---|---|---|---|
| E150 配方对齐消融（exact 架构补 per-link scaled 预训练） | `run_e150_v112_exactscaled.py` + `e150_jobs.sh`（pretrain×5 → cont×10 → eval → collect 四阶段） | `results/e150_v112_exactscaled/` | tab:conservation（exact-scaled col/E_total = 21.60/10.33（flow）、14.35/8.19（response），与 exact-unscaled 21.18/14.19、8.68 差 <0.5pp 且 E_total 略降 → 预训练损失不解释 exact-vs-direct 差距，配方混淆质疑关闭；penalty/projection 行复用 e146 冻结值由 collect 合并；守恒门 4.7e-11） |
| E151 matched-complexity 前沿（同段数预算） | `run_e151_v112_frontier.py`（E140 冻结 shards 重聚合，per-path 配对 bootstrap 5000；τ=5% 中位逐位复现发表值门 <0.02pp） | `results/e151_v112_frontier/` | tab:matched（SF 配对差 −2.05~−1.72pp CI 全不含 0；iso-error 11.0%→152 vs 117 段、8.8% response 可达而 flow 全 grid 不可达；AN 差 −0.14~−0.02pp 全不含 0） |
| E153 iso-segment 真匹配预算 | `run_e153_v112_isosegment.py`（逐 path L*=min(L_flow,L_resp) 网格插值；L* 超出任一方法支持区间的 path **排除而非端点钳制**，n_compared 逐容差报告） | `results/e153_v112_isosegment/` | tab:matched 的 L*/n 列 + 摘要 matched-segment 声明（SF n=96/96/96/95/14 iso 差 −2.02~−1.68pp CI 全不含 0；AN n=48/48/48/48/35 差 −0.16~−0.02pp 全不含 0；门 10/10） |
| E152 OOD response-generalization | `run_e152_v112_ood.py`（低载 [0.40,0.60]/高载 [1.35,1.55] 各 24 路径，冻结 e111 checkpoints 零重训；新增 unrestricted 闭式散度极值指标） | `results/e152_v112_ood/`（含 reference_high/low.npz） | §7 OOD 段（高载带 column/family/unrestricted 配对差 −1.31/−0.68/−5.30 全 CI 不含 0；低载带 family +0.68、E_total +0.57 显著、column +1.25 不显著——按评审口径如实报告 "largest gain inside and near the labeled domain"） |
| fig:pipeline 四面板总流程图 | `figures/src/fig0_pipeline_v112.py` | `figures/fig0_pipeline_v112.{pdf,png,svg}` | §1 fig:pipeline（替代原 Table 1/2；归因公式含 diag(∇T(f)) 算子；mini_axes 纵轴固定首次调用范围防单点重定标退化；t_j 画在 U_2 外满足 s∈U/t∉U） |

**P0 叙事与结构修订**：Abstract/Contributions/Conclusion 按"一个总命题"重写（三层 = principal structural / operational / learning）；八 OD 完整表 tab:cutspreadfull（e83 逐位 32 格）与三背景完整表 tab:bgfull（e79 A=41 点/B=11 点链口径）入附录、tab:cutspreadperod 删除；cor:mixedtol 正式化（可行性 iff max≤min；no-mixing 为额外报告规则 + hereditary→greedy 最优）；Table 11 加 Seg./chain 与 Total seg. 列；术语清理（account 名词 20+ 处清零、compact/balanced→absolute/mixed、美式拼写、network-distance progress）。

**论文模块**（paper_v112/，编译入口 `bash build_v112.sh`，54 页 0 undefined）：模块清单以 main_v112.tex 的 \input 链为准（connected_example、spatial_scan、results_main、results_anaheim、results_variation、results_adequate、results_pairs、anaheim_learning、appendix_proofs、proofs_connected、appendix_experiments 及其嵌套引用）；tab:learning-costs 表体本轮未改动，learning_protocol_v112 直接 \input 复用 `learning_cost_table_v111.tex`（快照内已含）。

历版更新说明：README_v111.md、README_v110.md、README_v109.md、README_v108.md、README_v107.md、README_v106.md、README_v103.md。
