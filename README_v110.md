# v110 配套复现包更新

v110 为 v109 外部评审（83/100 综合，TRB-B 匹配 87、专刊 "Mathematical Foundations for Trustworthy AI Applications in Transportation Systems" 匹配 93）后的实质修订版。论文主线整合为一条建设性方法链：**刻画何时可共享边界值 → 构造误差受控的最简空间剖面 → 把模型误差与压缩误差分开预算 → 用响应监督降低前者**。本轮六项问题全部落地：加权响应目标自有尺度归一化重跑（E139）、最终误差–分辨率扫描（E140）、Anaheim (25,2) 完整交通解释案例（E141）、增量流量误差（E142，并入 E140 表）、量化词措辞收紧与可选近似界的计算后弃用（E143）、九项一致性修复；摘要/贡献按评审给定版本重排，Table 3 实验地图改设置表，C.3/C.8 历史压缩，四张主文新图。

## v110 新增对象映射

| v110 对象 | 生成脚本 | 冻结输入/输出 |
|---|---|---|
| tab:fairmethods 加权行重写 + fig:v110pareto（新 Pareto 图）+ 协议节五尺度块（u_j/v_j/b_j/a_j/ā_j）与单位不变性段 | scripts/run_e139_v110_wnorm.py + e139_jobs.sh | **仅 wresponse 全 λ 网格重训**（e134 冻结初始化/1200 步/lr 3e-5/batch 32+8/rng 7102 其余全部不变），vscale=v_j=max(√mean‖D^S J^S‖²,1e-6)（D^S=TSTT 边际 t0·(1+0.75(F/cap)⁴)，E111 标签）；门：jscale 复现断言 + e134 精确中位合并。**尺度比 v_j/u_j=2.32–39.60（平方=有效正则失配 5–1568×，定量证实评审 P1 诊断）；λ*=0.1，final flow 0.0321/col 15.210/geo 6.932/bnd 12.862；path-level 改善 col 84/96、geo 79/96；λ∈[0.001,0.3] 全程可训练（旧共享尺度仅 λ≤0.003）**。results/e139_v110_wnorm/{selection.json, results.json, eval_wresponse_*.json, wresponse_*_*_cost.json}；35 个检查点（*.pt）不入包（脚本+种子再生，中位 ~155 s/次） |
| fig:v110resolution 误差–分辨率图 + tab:resolution τ 扫描表 + 跨网络结论句（"The relative importance of model and compression errors differs across networks..."） | scripts/run_e140_v110_resolution.py + e140_jobs.sh | τ∈{0.5,1,2.5,5,10}%，双网络双方法（e127/e130/e131 冻结列）；每行 E_summary/E_total/L/E_summary_ref_floor（对 aS 压缩的 oracle 地板）+ E_ΔF；τ=5% 行与 e127/e138 逐位门 1e-9 全过。**SF flow E_model 8.74、E_total 8.80(L233)→13.93(L77)；SF response 5.75、5.82→11.51（同段数全面更低：11.5% 需 77 vs 117 段）；Anaheim 0.94、1.06→8.14 与 oracle 地板（0.48→7.94）几乎重合（L 27.6→6.0，压缩主导）**。E_ΔF（E142 并入）：SF 5.14%/3.46%（绝对中位 13.6/8.8）、Anaheim 2.46%/2.08%（38.6/33.2）。results/e140_v110_resolution/{results.json, shards sf/anaheim × (method,seed)} |
| tab:anaheimcrossing（位置→跨界路段表）+ tab:anaheimsegmentsfull（(25,2) 完整四段输出表）+ Where 段落 + N_unique/N_chain/L_total 区分段 | scripts/run_e141_v110_anaheim_case.py | v105 代码路径重建 (25,2) 族（closure 断言）；crossing() 算前缀边界 signed C_e 贡献，位置 early/middle/plateau/destination。**早：链 231/34（269→261 +550.5）+271→192（+163.9）；中：405/4（191→190 +297.0）+66→65（+105.4）；平台 0.724–0.818：413-414/2（63→62 +5751.2）+87→86（−0.0）；终点：415/2（62→2 +2673.1）+2→87（−0.0）**；四段表 0.000–0.024/6/134.3/2.5、0.030–0.630/75/630.2/279.2、0.724–0.818/2/5751.2/0.0、0.911/1/2673.1/0.0；SF 链记账 N_unique 83–211、单链 L 8–14。results/e141_v110_anaheim_case/results.json |
| 近似公共值界（评审 §四可选，**计算后不纳入**） | scripts/run_e143_v110_bound.py | e112 21 态；全部 6 个 Anaheim 查询块路径=单原子 872/914 边，界/实际比 10.8–126（空转）→ 按评审决策规则弃用。results/e143_v110_bound/results.json（留档） |
| 四张主文新图 | figures/src/fig_{frontier,pareto,labelbudget,resolution}_v110.py | frontier 左面板改水平条形（评审 §七）；pareto x 轴 0.005 间隔三位小数刻度 + endpoint-cut 空心标记放大（视觉 QA 修复）；labelbudget 注明对数轴；resolution Anaheim 端 0.5% 标注上移避让。figures/fig_v110_*.{pdf,png} |
| 论文全文 | paper_v110/（main_v110.tex 及 \input 链 22 文件、references_v110.bib、build_v110.sh、cas-sc 三件套、TRB-MFAI_v110.pdf，49pp） | — |

## 关键协议要点

- **E139 冻结性**：重训范围=评审 P1 指定的最小集（仅 wresponse、仅归一化改变）；e111 预训练检查点、步数、批次、验证节奏、选择规则、96 测试路径全部不变；e134 其余五方法结果原样引用（results.json 由 e134 冻结行 + e139 新行合并，逐行核对）。
- **E140 口径**：统一归一与 e127/e138 同管线（逐行 E/fam_scale_reference→path 均值→路径中位×100）；E_summary_ref_floor 为 oracle 压缩地板（对 aS 同容差压缩）；两方法共用同一参考尺度，图中不混用分母。
- **E141 位置先验**：四个展示位置按自由流进度先验指定（early<0.25、middle 0.3–0.55、平台取实测高原、destination=最后链），非按归因值挑选；平台段 0.724–0.818 的两条链贡献全部 5751.2。
- **量化词（P4）**：全文统一 "guaranteed uniformly over all flow-conserving responses"（强量词）与 "path-feasible response envelope"（模型可实现包络），不再出现 "realizable response space(s)"。
- **九项一致性（P6）**：C.5 baseline=[-13,502, 24,260]（[-13,024,24,064] 为 θ=0.5 条件已标注）；Eq C.1/Table C.5 引用修复；完整分段表落主文；Anaheim 协议句改"同参数化与评估原则+网络特定设置"；Table 12 列名 Family boundary error；24-path timing subset 标注；六维模型↔八维零背景模型对应段（C.3：结构/训练域/预训练/检查点）；label-cost 句改"较少标签恢复较多改善"；数据可用性=GitHub+Zenodo 实址。

运行布局与历史版本一致：scripts/ 置于 experiments/，results/ 置于 experiments/results/，网络输入在 data/，绘图脚本在 figures/src/（按含 experiments/results 的祖先目录自动解析仓库根）。编译入口 bash build_v110.sh（在 paper_v110/ 内）。E139 为完整再生路径（35 次续训，-P16 多进程约 25 分钟）；E140/E141/E143/E144 为冻结数组再计算（分钟级，E140 含分片并行；E144 须先于 E140 collect 运行，collect 断言其存在）。

## 审稿轮同步（codex gpt-6.1-sol，2026-11-01）

三轮收口 7.0 → 9.0 → **10/10 ready**：R1 修复 Pareto 全 λ 曲线合并（e139 collect 并入 e134 冻结扫描）、新增 **E144 参考剖面段数**（L_ref：SF 233.0/199.8/149.0/116.7/76.7、Anaheim 25.92/20.0/12.0/9.75/5.92；E140 collect 自行合并、fig × 标记画自身段数、论文改称 reference compression）、压缩主导结论限定到工作容差、成本表引用/计时口径、设置表拆行、平台第二链路辨析；R2 修复 E144 生成流程接入与 24 路径子集合计口径（≈0.3 s/路径）。本包脚本与结果均为终版。
