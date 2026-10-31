# v106 配套复现包更新

v106 为外部评审（84/100, focused major revision）后的结构大修版：connected-region 分类定理提升为主定理（Theorem 1），实验章重组为三问结构（6.1 何时单一摘要够用 / 6.2 什么产生空间变化 / 6.3 完整剖面与紧凑摘要），学习实验测试路径从 24 条扩展到 96 条并补齐配对 bootstrap 与 leave-one-path-out 不确定性，新增容差灵敏度 frontier（τ=2.5–10%）与理论不平衡量 ‖d_j‖₂—实际空间范围联系。学习结论不变且加强：column 误差 21.18%→14.19%、geometric 8.74%→5.75%，92/96 路径改善。

## v106 新增对象映射

| v106 对象 | 生成脚本 | 冻结输入/输出 |
|---|---|---|
| 表 tab:pathtable（96 路径×4 组合并学习主表）、摘要/正文 21.18/14.19/8.74/5.75、bootstrap CI、LOO 区间、92/96 与 90/96、system 中位数 2.52/3.13 | scripts/run_e119_v106_expanded_paths.py | 模型与验证选择冻结自 results/v104_learning/results.json 的 selection 字段；e111_v102 检查点与 v104 数据集；results/e119_v106_expanded_paths/results.json |
| 原 24 路径子集响应主导分解（256.1/4.1、158.1/3.8） | results/v104_learning/results.json（summary.flow / summary.response 的 response_norm、weight_norm） | e111_v102 四方法协议 |
| 表 tab:tolerance（τ=2.5/5/7.5/10% frontier）、正文 2.7–10.4 / 4.7–18.8 倍容差句 | scripts/run_e120_v106_tolerance.py | e114_v103（level 1.0, halfwidth 0.2）+ e116_v105_loadlevels（0.7/1.3）+ v105_anaheim_profile/profile.json；零新求解；results/e120_v106_tolerance/results.json |
| §6.1 e121 段（‖d_j‖₂ 中位 1196.4/4803.2/15763.3、13.2 倍、Spearman 0.99、S_j/‖d_j‖₂=2.53 vs √22=4.69、归一化误差相关 0.53） | scripts/run_e121_v106_imbalance_link.py | e114_v103 + e116_v105_loadlevels 账户与梯度数组；results/e121_v106_imbalance_link/results.json |
| 图 fig:screenlines（竖排 Upstream/Downstream screenline 对比） | figures/src/plot_screenline_v106.py | e107_v100_screenline.json（区域与账户）+ e86_v84_newod.npz（C 列）；figures/fig_screenlines_v106.pdf |
| 图 fig:commonsummary（误差排序单栏图）与附录 fig:coverageintervals（完整区间图） | scripts/plot_v106_common_accounts.py | v104_spatial/coverage.json（halfwidth 0.2 行） |
| 图 fig:anaheimprofile（(25,2) 剖面+四部分分解两面板）与附录 fig:anaheimrest（其余五查询） | scripts/plot_v106_anaheim_profile.py | v105_anaheim_profile/profile.json |
| 论文全文 | paper_v106/（main_v106.tex 及 \input 链、references_v106.bib、build_v106.sh、TRB-MFAI_v106.pdf） | — |

## e119 协议要点（复现门内置）

- 96 条测试路径 = 原 24 条 + 新路径种子 102113 下同规则生成的 72 条（1/2/3/6 active queries 各 18 条新路径）；生成规则与 e111 完全一致（基线 ×U(0.65,1.3)、active 查询取循环规则 [(r+i)%6 for i in range(k)]、Δ=U(0.15,0.3)、r 奇数时交替变号、41 态）；bootstrap 重采样种子 20261030（10,000 次），与路径生成种子不同。
- 模型、训练数据、验证选择全部冻结：selection 从 v104_learning/results.json 读取（step==0 取 CKPT_DIR/seed{seed}.pt，否则 e111_v102/{method}_{seed}.pt）。
- 守恒门包含 522 条背景 OD（load_sioux_falls_trips 的 Mb、qbgv）；复现门对齐 v105 发表值（flow column 21.61→重算 21.6135 等，abs_dev < 0.005）。
- bootstrap 10,000 次重采样，种子 20261030；path 为独立单元（active queries 与 seeds 先在路径内平均）。
- 12 MB 的 paths_new.npz（96 条路径完整需求矩阵）不入本包：由 protocol.json 的 NEW_SEED 与上述生成规则可逐位重建。

运行布局与历史版本一致：scripts/ 置于 experiments/，results/ 置于 experiments/results/，网络输入在 data/；训练检查点保留在原项目 experiments/results/（体积原因不入本包）。编译入口 bash build_v106.sh（在 paper_v106/ 内）。e120/e121 为零新求解的存储数组再计算，运行时间分钟级；e119 需加载冻结检查点，单卡约 1 小时。

本包为本地同步版本；公开版本化归档（GitHub + DOI）以上传为准，论文可复现性声明以实际归档地址为准。
