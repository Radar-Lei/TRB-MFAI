# v111 配套复现包更新

v111 为 v110 外部评审（87/100 综合"强稿，聚焦型大修"）后的身份重构版。论文从"一个边界值何时失效"反转为**以边界归因剖面为原始解释对象**：拓扑定理刻画其何时精确退化为单一可迁移值；一般情形返回指定误差下的最简空间剖面；学习型分配用响应监督提高剖面准确性。本轮六项核心修改全部落地（剖面优先叙事、可信性三层含义进引言首页、学习型剖面修复核心图、balanced(1%+10%) 容差为推荐默认 / compact(5%) 为高压缩档、六目标评测降附录、开发史语言清除），另有符号体系向量优先化（g_j/c_j/𝒢）、Example A/B 拆分、四组新实验 E145–E148、增量流量误差并入主表。

**新实验与对象映射**

| 实验 | 脚本（experiments/） | 结果 | 论文位置 |
|---|---|---|---|
| E145 学习剖面修复案例（预指定中位改善选例） | `run_e145_v111_repair.py` + `fig_v111_repair.py` | `results/e145_v111_repair/` | §7 fig:repair 段落（path 22 / OD (7,12)：最坏边界 369.5→80.1（该处 −78%）、族最坏 369.5→115.2、E_total 679.1→463.7、段 186→190；ff 600.6/cong 5018.7/总 5619.3） |
| E146 守恒方式消融（exact / penalty / projection × flow / response） | `run_e146_v111_conservation.py` + `run_e146b_v111_reselect.py`（统一选择规则含 update-0 重评） | `results/e146_v111_conservation/` | 附录 app:conservation + tab:conservation（exact 21.18/14.19；penalty 183.7/177.9（输出残差 42.1、导数残差 1.16、积分归因残差 R_j 0.66）；projection 104.1/99.4（守恒 2e-11、R_j 4e-15）；选择规则含初始模型（E146b 重选审计）） |
| E147 标签预算随机子集重复（3 个 query-balanced 嵌套子集/预算） | `run_e147_v111_labelrep.py` + `fig_v111_labelbudget.py` | `results/e147_v111_labelrep/` | §7 标签效率段（阈值留正文）+ 附录 app:labelbudget fig:v109labelbudget（36→16.35 … 384→14.19；分层聚合：重复内 96 路径中位→三重复中位；IQR ≤0.8pp；50%@36/80%@144/90%@288，族 80%@72） |
| E148 拓扑分布分位数 + 等时钟比较（只读冻结数组） | `run_e148_v111_light.py` | `results/e148_v111_light/` | §4 拓扑释放段 + tab:topologydist；§7 训练预算与时钟段（SF 0.070/0.124 s/update、流总时处 response K_mse 2.99× 更低；AN 0.539/1.774、2.19× 流总时起占优） |
| Anaheim 增量流误差（路径中位聚合） | e129 数据 + e130 检查点（见 e145 目录 anaheim_increment.json） | 同左 | tab:pathtable Anaheim 增量列（2.46/2.03） |

**协议要点**：E146 直连 MLP 宽度由参数方程解出与精确参数化对齐（714,979 vs 715,040），预训练 per-link scaled MSE + 末层偏置初始化为训练均值流场（raw-MSE 会发散）；E147 子集序 rng(20261101+ρ+97j)；E148 无再训练。全部实验沿用既定约定：单线程 × 多进程、冻结检查点续训、E127 的 greedy_segments/learned_columns 复用。

**论文模块**（paper_v111/，编译入口 `bash build_v111.sh`，51 页 0 undefined）：新模块 `conservation_ablation_v111.tex`、`label_budget_protocol_v111.tex`；`learning_legacy_results_v111.tex` 承接自主文移入的六目标表（tab:fairmethods）、Pareto（fig:v109pareto）与收敛图（fig:v109convergence）；tab:pathtable 为合并主表（总流/增量/列/族/最终 summary/段数）。

历版更新说明：README_v110.md、README_v109.md、README_v108.md、README_v107.md、README_v106.md、README_v103.md。
