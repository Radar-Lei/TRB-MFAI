# v108 配套复现包更新

v108 为 v107 外部评审（82/100 综合，六项实质修改清单）后的大修版，主线整合为一条：**刻画何时可共享边界值 → 构造误差受控的最简空间剖面 → 说明并改善学习模型生成这些解释的准确性**。六项修改全部落地：§3.2 表重组（边界族 × 响应类 × 解释精度）+ 精确族判据命题；§5.3 总误差预算（ε_summary + δ_j 三角不等式链）+ 最少分段摘要前移 + Procedure 1 完整用户管线；§6.1 逐方向排序分段 + Screenline A/B + 5% 绝对容限措辞 + 混合容限敏感性；§5.6 训练目标/守恒参数化/误差分解入正文 + 96 路径四方法统一比较 + 五种子配对表；§7 交通贡献围绕"空间响应如何变化"重写；三贡献聚焦。新增第二网络学习案例（Anaheim，同守恒参数化，48 测试路径）与逐方向容限扫描。

## v108 新增对象映射

| v108 对象 | 生成脚本 | 冻结输入/输出 |
|---|---|---|
| 逐方向边界族共享模块（36 投影法向、嵌套链、两侧连通过滤、跨方向去重） | scripts/fam_v108.py | e126/e127/e133 共用；门：方向族掩码并集 == e114 族 |
| §6.1 逐方向分段 + tab:tolerance（逐方向版）+ Anaheim 段表 tab:anaheimsegments + Figure 5 段叠加 | scripts/run_e126_v108_persdirection.py | SF = e114 冻结账户（零新求解）；Anaheim = v105_anaheim_profile/profile.json；results/e126_v108_persdirection/（含 segments_detail：段值 β、ε、progress 区间） |
| tab:totalerror（总误差预算）+ §7 响应主导段 + eq:coldecompmain 的实证支撑 | scripts/run_e127_v108_totalerror.py | 96 路径（e111+e119 冻结选择与参考列）；方向族；E_model/E_summary/E_total/E_single/E_equal + 分解闭环门 ≤6.4e-12、E_summary≤ε^L 与 E_total≤预算逐行断言；results/e127_v108_totalerror/{results.json, rows.json}（2880 行）。**表值口径（v108.1 统一）：逐行 E/fam_scale_reference 归一，path 内平均，96 路径取中位（与 tab:pathtable geometric 列同管线）** |
| tab:fourmethods + tab:fiveseed（96 路径四方法统一比较） | scripts/run_e128_v108_fourmethods.py | 冻结选择（v104_learning selection），无再训练；复现门 vs e119 全列 abs_dev=0.0；results/e128_v108_fourmethods/ |
| Anaheim 学习参考数据（流态/导数中心/测试路径） | scripts/run_e129_v108_anaheim_data.py + e129_jobs.sh | 种子 108001/108002/108003；2488 次带灵敏度求解（~35 s/态）；门：守恒 9.09e-12、完备性 1.2e-5；**本包仅含 meta/demands/families/collect，38 个求解分片（196M）与 setup.pkl 留原项目可由脚本再生**；results/e129_v108_anaheim_data/ |
| Anaheim 训练（两阶段：流预训练 + 方法目标微调） | scripts/run_e130_v108_anaheim_train.py + e130_jobs.sh | AnaheimNet 6→128×3→1406 softmax×14060 路径；stage A 4000 步 + stage B 1200 步；K-MSE 验证选择（流门槛 1.1×）；**本包仅含 pre_seed*.json 与 *_cost.json 曲线，检查点（1.7G）留原项目** |
| tab:anaheimlearning（第二网络学习案例）+ Anaheim 总误差 | scripts/run_e131_v108_anaheim_eval.py | e130 冻结选择，48 测试路径；flow 1.64%/response 1.31% 列误差、E_model 17.0→13.1、改善 44–48/48；results/e131_v108_anaheim_eval/ |
| supp:tolerance 混合容限段 | scripts/run_e132_v108_mixedtol.py | 零新求解（v105 Anaheim 账户）；ε_abs=1%+τ=10% → 小位置误差中位 0.55→0.28、段数 10→12；results/e132_v108_mixedtol/ |
| tab:tolerance 逐方向列 + Figure 2 Panel B | scripts/run_e133_v108_pertol.py | 零新求解（e114/e116 + e126 方向链）；τ∈{2.5,5,7.5,10}%；results/e133_v108_pertol/ |
| 图 fig2 双面板（(a) 单值适用性 (b) 逐方向容限 frontier） | scripts/fig2_frontier_v108.py | Panel A = v104_spatial/coverage.json；Panel B = e133；figures/fig_v108_summary_frontier.pdf |
| 图 fig5 旗舰应用图（剖面 + teal 段阶梯 + ±ε 带 + 分解） | scripts/fig4_anaheim_v108.py | v105_anaheim_profile/profile.json + e126 segments_detail（query (25,2)）；figures/fig_v108_anaheim_profile.pdf |
| 论文全文 | paper_v108/（main_v108.tex 及 \input 链 22 文件、references_v108.bib、build_v108.sh、highlights_v108.tex、TRB-MFAI_v108.pdf） | — |

## 关键协议要点

- **e127 总误差口径（v108.1 统一）**：fam_scale_reference = max_U |a^S_j(U)|（参考族尺度）；表值为逐行归一后 path 内平均、96 路径中位；段数按 distinct direction chains 去重。E_single/E_equal 对照分别为全族单一共同值与同段数等距分段。
- **e129/e130 体积约定**：求解分片与训练检查点不入本包（复现路径：e129_jobs.sh 12 workers ≈ 1 h + e130 训练 ≈ 40 min/seed 预训练 + 5–9 min/(method,seed) 微调）；协议、种子与门全部在脚本与 protocol 文件中。
- **Anaheim 全部响应 checkpoint 落 1200 步上限**（budget-limited 而非 validation-plateaued），论文 §C.12 如实注明；结论跨种子一致。

运行布局与历史版本一致：scripts/ 置于 experiments/，results/ 置于 experiments/results/，网络输入在 data/。编译入口 bash build_v108.sh（在 paper_v108/ 内，需 cas-sc.cls 三件套同目录）。e126/e127/e128/e131/e132/e133 均为冻结数组再计算（零均衡重解、零重训），单机分钟级；e129/e130 为完整再生路径（小时级，单线程×多进程）。

本包为本地同步版本；公开版本化归档（GitHub + DOI）以上传为准，论文可复现性声明以实际归档状态为准（当前尚未公开，论文已如实表述）。

## v108.1 审稿修正同步（2026-10-31，codex round-1 后）

- **run_e130_v108_anaheim_train.py 修正**（blocker）：validate() 改为 K(D^L J^L − D^S J^S)（原实现误用 f·t(f) 权重且未减参考响应），权重统一为 TSTTT 边际 t0(1+0.75(F/c)⁴)（原 0.15 为 BPR 成本系数，非 Σf·t 的梯度系数）。训练损失不变（种子固定、轨迹确定）；10 个微调重跑后选择仍全部落 1200 步上限，选中权重与修正前一致，e131 评估值逐位复现。
- **run_e131 重算**：overall 与表值不变；five-seed path 级改善 column 46–48/48、geometry 44–47/48；E_total 配对改善（path–seed 单元）73.3%。
- **tab:totalerror 口径统一**（major）：改为主协议逐查询归一（表值 8.74/4.76/11.35/34.3/19.7 与 5.75/4.75/8.68/33.0/17.2，段数按 distinct chains 去重 117/117）；"摘要项主导"结论改为"flow 下模型项主导、response 后两项相当"。
- 图 fig_screenlines_v108 重导出（图内 Upstream/Downstream → Screenline A/B，视觉 QA PASS）；fig3_screenlines_v108.py 入 scripts/。
- 检查点可用性声明改写（脚本+种子可再生、索取可得）。

## 公开归档（2026-10-31 已发布）

- GitHub：https://github.com/Radar-Lei/TRB-MFAI（public，分批推送完整内容）
- Zenodo DOI：10.5281/zenodo.23831225（https://zenodo.org/record/23831225，tarball 153,884,826 字节，MD5 ab3f545f20bf01d700e29a96fb33068d）
- 论文 Code and data availability 段已更新为上述地址（v108 终版）。
