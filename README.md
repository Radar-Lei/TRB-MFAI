# v102 本地配套包更新

新增连通区域分类完整循环空间结构例、端点监督四方法五种子24独立路径、Anaheim全部1406正需求的空间计算。新边界训练仍无整体优势；论文主体为分类、计算与交通机制。

新脚本为scripts/run_e111_v102_pairs.py、run_e111_v102_connected.py、run_e112_v102_anaheim.py和report_e112_v102.py。评价、标签、冻结协议与成本在results/e111_v102/；Anaheim原始数组在results/e112_v102/，网络输入在data/Anaheim/。完整训练与预算检查点保留在原项目experiments/results/e111_v102/，没有重复复制到本包。

原有依赖及e93初始模型继续沿用项目；将scripts、results、data分别放回experiments对应子目录运行。编译入口bash paper/build_v102.sh。此包已在本地同步，尚未公开发布。

---

# v101 本地配套包更新

新增e108直线截面五种子对照、e109三方法固定步数延续训练、e110验证集校准与全新路径测试。两轮结构目标均无整体优势，具体结果请见项目根目录NARRATIVE_REPORT_v101.md。本地包未公开发布，不冒称已有持久归档。

新增脚本位于scripts/，原始数值数组和JSON位于results/e109_v101、results/e110_v101。恢复运行仍沿用下文原有布局：scripts内容置于experiments/，results置于experiments/results/，网络文件置于experiments/data/；原e93数据集和五个检查点在原项目experiments/results/内。全部训练检查点保留在原项目目录，本同步包保留评价数组与选择记录。

第一协议同为600更新步，不是同墙钟成本。第二协议另有三个候选的验证搜索，并保留首轮负面结果。新旧路径完整逐种子、逐查询结果见JSON的rows及evaluation数组。

---

# TRB-MFAI v99 配套复现包

本包在本地整理，尚未提供公共仓库地址。脚本、结果数组和训练检查点可向通讯作者索取。下文 v94/v87 段落为历史说明，当前对象编号以本表为准。

| v99 对象 | 脚本或文件 | 冻结输入/输出 |
|---|---|---|
| 表2：A/B/C范围；表C.2：逐OD绝对分母 | scripts/run_e106_v98_summary.py | e85_v84_precision.json、e84_v83_coverage.json；e106_v98_summary.json |
| 表3：六查询范围、负占比与邻域账户 | run_e87_v85_newod_refine.py、run_e103_v94_negative_share.py、run_e99_v91_predefined_boundaries.py | 对应e87、e103、e99结果 |
| 表4：固定负极值边界的四项分解 | scripts/run_e106_v98_summary.py | e87_v85_newod_refine.npz、e91_v87_negative_mechanism.json；e106_v98_summary.json |
| 表5：代理边界误差绝对量及原百分比 | scripts/run_e106_v98_summary.py、run_e93_v89_fullbg_surrogate.py | e93_v89_fullbg_surrogate.json；e106_v98_summary.json |
| 表C.3：背景/研究OD节点散度 | run_e83_v82_validation.py | e83_v82_validation.json |
| 图2：boundary-dependent account | figures/src/fig_resp_field_v98.py | fig_resp_field_v30.npz；figures/fig_resp_field_v98.pdf |
| 图3：仅标关键节点、网络放大、合计分行 | figures/src/fig_negmech_v99.py | e94_v89_divsplit_links.json；figures/fig_negmech_v99.pdf |

v99为文字与图形修订；单需求的其余527需求均固定公开值，直接与学习设置一致，无需重跑。v98重汇总不求解均衡、不训练、不重新选择边界。运行时沿用原有项目结构：将 scripts 内容置于 experiments/，results 置于 experiments/results/，网络数据置于 experiments/data/；figures/ 放在同一项目根目录下。

---

# TRB-MFAI v94 可复现发布包

> **v94 更新（2026-10-27）**：本包已同步至论文 v94。新增/更新对象映射：
> | 论文对象 | 生成脚本 | 结果文件 |
> |---|---|---|
> | Table 4 负占比列 r_j⁻ 与 Predefined 列（六查询） | run_e103_v94_negative_share.py | e103_v94_negative_share.json |
> | Table 4 单 OD 复算（C_j 与连通区间，见附录 C.5） | run_e95_v90_singleod.py + run_e97_v91_singleod_identity.py | e95_v90_singleod.json/.npz, e97_v91_singleod_identity.json/.npz |
> | Table 5 自由流–拥堵分解 | run_e101_v93_ffcong.py | e101_v93_ffcong.json/.npz |
> | Table 6 全背景代理验证 | run_e93_v89_fullbg_surrogate.py | e93_v89_fullbg_surrogate.json |
> | §7.4 负机制跨界链路（图 3 侧栏） | run_e94_v89_divsplit_links.py | e94_v89_divsplit_links.json |
> | §7.5 ε^cut/span 与跨查询、全背景六查询误差族 | run_e93_v89_fullbg_surrogate.py（全背景六查询）；run_e92_v88_cut_error.py（零背景八查询） | e93_v89_fullbg_surrogate.json/.npz, e92_v88_cut_error.json |
> | C.3 小带宽学习评估 | run_e102_v93_smallband_learned.py | e102_v93_smallband_learned.json |
> | C.3 增量流误差 E_ΔF | run_e98_v91_incremental_flow.py | e98_v91_incremental_flow.json |
> | C.4/C.5 模型稳健（20 路径、θ 扫描；六查询协议复核） | run_e84_v83_coverage.py（八查询）；run_e90_v87_settingcheck.py（六查询） | e84_v83_coverage.json/.npz, e90_v87_settingcheck.json |
> | C.5 预定义边界族（三跨零 OD） | run_e99_v91_predefined_boundaries.py | e99_v91_predefined_boundaries.json |
> | C.5 小带宽直接复算 | run_e100_v92_small_demand.py | e100_v92_small_demand.json/.npz |
> 代理训练检查点（e93 五种子 full-bg + 早期 zero-bg）可向通讯作者索取（体积原因不入本目录）。

# （以下为 v87 版基础说明，对象编号以 v87 论文为准；v94 对象映射见上表）

本目录为自包含复现包：每个表格/图给出对应生成脚本与结果数组。网络数据在
`experiments/data/SiouxFalls_{net,trips,node}.tntp`（公共 TNTP 仓库同源文件）；
运行时将 `scripts/` 与 `results/` 置于 `experiments/` 同级结构，或直接在仓库
根目录执行（脚本按相对路径读取）。

## 环境

Python ≥3.10，numpy / scipy / torch（仅代理检查用）。项目根目录 `.venv` 已配置；
`requirements` 清单见 `repro/requirements-v82.txt`。BLAS 单线程限制由脚本自动设置。

## 表 / 图 → 生成路径

| 论文对象 | 生成脚本 | 结果文件 |
|---|---|---|
| Table 1（平衡层级） | 解析定义（无计算） | — |
| Table 2（三设定割范围主表） | gen_v83_tables.py（B/C 列）+ e83（SUE A、learned）+ e79/e78（基准） | e83_v82_validation.json/.npz, e84_v83_coverage.json, e79_bgreroute.json, e78_sue_allod.json |
| Table 3（study/background 节点分解） | run_e83_v82_validation.py | e83_v82_validation.json:per_od[OD].signed_node_decomposition |
| Table 4（{10}→{10,17} 边界分解 + Rel. 列） | run_e84_v83_coverage.py；Rel. 列与 A 侧对照 = run_e85_v84_precision.py | e84_v83_coverage.json:C_k10_theta1；e85_v84_precision.json:boundary_A_vs_C |
| Table 5（六个不含节点 10 的查询） | run_e86_v84_newod.py | e86_v84_newod.json/.npz |
| Figure 1（block-cut 合并示意） | figures/src（绘图源） | fig2_block_cut.pdf |
| Figure 2（三类权重响应场） | experiments/figdata_resp_field_v30.py | fig_resp_field.pdf；fig_resp_field_v30.npz（仓库内） |
| Figure B.1（tanh/sin 反例验证） | run_e4_postcheck.py 系 | fig2_tanh.pdf |
| 附录 C 精度表（B/C 细化、θ、路径集） | run_e83/e84_v83_coverage.py | e83_v82_validation.json, e84_v83_coverage.json |
| 附录 C 近零 41/81/161 精度段 | run_e85_v84_precision.py | e85_v84_precision.json:precision.near_zero_rows |
| 附录 C 9.56% 定位句 | run_e85_v84_precision.py | e85_v84_precision.json:jacobian_error_localization |
| 附录 C.2 误差指标定义段 | 与 e83/e9b 计算实现一致（column_error 分母 = 直接 SUE 列二范数） | e83_v82_validation.json:surrogate_zero_background |
| 附录 C.5 区间表（三高百分比查询） | run_e87_v85_newod_refine.py（式(5)闭式 + 连通枚举） | e87_v85_newod_refine.json:per_od[OD].interval_41 |
| 附录 C.5 41→81 细化段 | run_e87_v85_newod_refine.py | e87_v85_newod_refine.json:per_od（column/phi_s/S/范围变化） |
| 定理 5/6、Remark（√ε 阶） | 解析构造（论文内自足） | — |

## 复现命令（在仓库根目录）

```bash
.venv/bin/python experiments/run_e85_v84_precision.py   # ~1 s
.venv/bin/python experiments/run_e86_v84_newod.py       # ~8 s
.venv/bin/python experiments/run_e87_v85_newod_refine.py # ~8 s（41→81 细化 + 区间）
.venv/bin/python experiments/run_e83_v82_validation.py  # ~50 s（B/C 主链）
.venv/bin/python experiments/run_e84_v83_coverage.py    # ~29 s（路径集/θ 覆盖）
bash paper/build_v85.sh                                  # 编译 TRB-MFAI_v85.pdf
```

门控断言内置于各脚本（守恒、可加性、冻结记录复现等），失败即抛错中止；
`experiments/results/*_console.log` 为实际执行日志。数值溯源逐项登记于
`NUMERIC_PROVENANCE.md`（v84 段）。

## 代理模型检查点

learned 列所需的五个检查点在 `experiments/results/e5_ckpt/`（仓库内，未收入本
包以控制体积；训练协议见论文附录 B.3 与 `experiments/run_e5.py`）。

## v87 新增对象

| 论文对象 | 生成脚本 | 结果文件 |
|---|---|---|
| Table 5 Connected interval 列 | 已发布端点整数化（e87） | e87_v85_newod_refine.json:per_od.interval_41.connected |
| §7.6 负账户机制段 + 附录 C.9 表 | run_e91_v87_negative_mechanism.py | e91_v87_negative_mechanism.json:mechanism |
| §7.6 设定检查段 + 附录 C.10 表 | run_e90_v87_settingcheck.py | e90_v87_settingcheck.json:conditions/baseline |
| 附录 C.5 第一/三/四检查段 | run_e87_v85_newod_refine.py / e91 / e90 | 对应 JSON |
| Table C.3 C_j 列 | run_e91_v87_negative_mechanism.py | e91_v87_negative_mechanism.json:eight_query_Cj |
