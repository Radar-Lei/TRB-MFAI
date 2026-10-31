# v107 配套复现包更新

v107 为 v106 外部评审（8.8/10 综合科学质量，focused major revision）后的正面叙事聚焦版：摘要/贡献/引言首段换评审给定版本（主线"每个查询都得到带误差控制的公共摘要或空间剖面"），四层 transferability 层级表进正文并给出数值诊断，§6.1 改名自适应选择，§7 改名 Theory-informed learning，新增三项 A 级实验（bound-chain 闭环 / attribution-path 稳健性 / 受限响应空间诊断）与一项 B 级拓扑统计。学习结论不变：column 21.18%→14.19%、geometric 8.74%→5.75%，92/96 与 90/96 路径改善；新增 **E_all/B 中位 0.95——理论积分界是已实现最坏边界误差的紧致预测器**。

## v107 新增对象映射

| v107 对象 | 生成脚本 | 冻结输入/输出 |
|---|---|---|
| 表 tab:pathtable 的 Unrestricted/Bound 列与正文 bound-chain 段（E_all/B 中位 0.95/0.95、链恒等式 1.6e-15）、E_geo/E_all 中位 0.51/0.48 | scripts/run_e122_v107_boundchain.py | 96 路径 = results/e111_v102/paths.npz（24，含 J 灵敏度）+ results/e119_v106_expanded_paths/paths_new.npz（72，含 J；分组 = 两 npz 的 group 拼接）；学习模型 jvp 用 e111_v102 检查点（v104_learning/results.json selection 冻结）；参考列一致性数组 = results/e119_v106_expanded_paths/reference_columns.npz；results/e122_v107_boundchain/{results.json, rows.json} |
| 表 tab:pathrobust（22 顺序六行表）、附录 C.11 per-order 记录段 | scripts/run_e123_v107_pathrobust.py | 直线基准 = e87 冻结 41 点直线数组；U_- 区域 = e91 冻结；置换种子 20261031；每段 41 态；门：直线复现 1.7e-15 / 端点 6.1e-9 / completeness 1.5e-7；results/e123_v107_pathrobust/results.json |
| 表 tab:levelsdiag（层级数值诊断）+ 三水平稳定句 + Anaheim 值 | scripts/run_e124_v107_respspace.py | synthetic = nets.two_bridges post-merge 权重；Sioux = setting-C 路径结构 5280 路径（e109 setup）；Anaheim = e112 21 态中点；SVD 响应空间基（rank 截断 1e-9 相对）；results/e124_v107_respspace/results.json |
| §4.2 B2 四网络句（SF/Anaheim/Winnipeg/EMA atom 统计与中位约束份额） | scripts/run_e125_v107_topologystats.py | 公开 TNTP 拓扑（候选网 data/candidates/：EMA、Barcelona、Winnipeg net+trips）；纯拓扑预处理零求解；results/e125_v107_topologystats/results.json |
| 图 fig1（(a) 网络+block-cut 树 / (b) 查询类合并双面板） | scripts/fig1_block_cut_v107.py | nets.two_bridges() 构建时断言（cuts=[1,2,3,5,6]、6 atoms、2 classes）；figures/fig1_block_cut_v107.pdf |
| 图 fig2 双面板（18 查询误差排序 + τ–段数 frontier） | scripts/fig2_frontier_v107.py | v104_spatial/coverage.json（误差排序）+ results/e120_v106_tolerance/results.json（frontier）；figures/fig_v107_summary_frontier.pdf |
| 图 fig3 左右两栏 screenline | scripts/fig3_screenlines_v107.py | results/e107_v100_screenline.json + results/e86_v84_newod.npz 的 C 列；上下游判定 = 分离线与 s–t 直线段交点进度（0.608/0.785，运行时 assert + revision-stage/v107/SCREENLINE_LINKS.json 证据）；figures/fig_screenlines_v107.pdf |
| 图 fig4（(25,2) 剖面+四部分分解，显式线型×marker 编码）与附录五剖面 | scripts/fig4_anaheim_v107.py | results/v105_anaheim_profile/profile.json（数据未变）；figures/fig_v107_anaheim_profile.pdf + fig_v107_anaheim_rest.pdf |
| 附录区间图 | （与 v106 相同数据） | fig_v107_coverage_intervals.pdf = fig_v106_coverage_intervals.pdf 原样复制（内容未变） |
| 论文全文 | paper_v107/（main_v107.tex 及 \input 链 19 文件、references_v107.bib、build_v107.sh、highlights_v107.tex、TRB-MFAI_v107.pdf） | — |

## e122/e123 协议要点（复现门内置）

- e122：学习列由模型 jvp 逐状态计算，参考列与参考灵敏度取 e119 冻结数组（同状态同查询）；K_j 为边界半范数（终点差+内部散度），B_j = |Δq_j|·trapz(K_j(h_j(λ)))；归一分母 Kr = K_j(参考列)。门：链恒等式相对违反 ≤1e-6（实测 1.6e-15）、24 路径 paper 值复现 ±0.02（21.18/14.19/8.74/5.75）。
- e123：22 顺序 = canonical、reverse、20 固定种子置换（20261031）；每 OD 一段 41 态逐段推进；直线基准复用 e87 冻结数组（复现门 1.7e-15）；机制四分解在固定 U_- 上、ub=u@B 预计算；门含端点偏差 6.1e-9 与 completeness 1.5e-7。
- e124：响应空间基用 SVD（U 列空间、rank = 奇异值 > 1e-9·σ₀；QR 无列主元版本已弃用——误判秩 34≠53）；η_conn 掩码只统计查询 OD 块路径经过的 atoms；Anaheim 用 e112 21 态中点状态。

运行布局与历史版本一致：scripts/ 置于 experiments/，results/ 置于 experiments/results/，网络输入在 data/；训练检查点保留在原项目 experiments/results/（体积原因不入本包）。编译入口 bash build_v107.sh（在 paper_v107/ 内，需 cas-sc.cls 三件套同目录）。e122–e125 均为冻结数组再计算（零均衡重解、零重训），单机分钟级。

本包为本地同步版本；公开版本化归档（GitHub + DOI）以上传为准，论文可复现性声明以实际归档地址为准。
