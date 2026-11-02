# TRB-MFAI 复现包（当前版本 v112）

当前投稿版本为 **v112**（2026-11-02，v111 外审 8.4/10 贡献聚焦型大修）：论文 "Topology-aware spatial attribution in traffic assignment: Structural guarantees, adaptive summaries, and response-supervised learning"。本轮收拢为一个总命题、三层贡献（principal structural / operational / learning），AI 主线贯通（fig:pipeline 四面板总流程图替代 Table 1/2），新增配方对齐消融（E150）、matched-complexity 前沿（E151）、OOD 泛化（E152）、iso-segment 真匹配预算（E153），八 OD 完整表与三背景完整表入附录、cor:mixedtol 正式化、术语全面清理，对象映射与协议要点见 **README_v112.md**；论文全文在 paper_v112/（编译入口 bash build_v112.sh）。

目录结构：scripts/（实验与绘图脚本，置于 experiments/ 运行）、results/（冻结结果 JSON，置于 experiments/results/）、data/（网络输入）、paper_vXXX/（历版论文快照）。历版更新说明：README_v111.md（v111 身份重构）、README_v110.md、README_v109.md、README_v108.md（v108 六项大修）、README_v107.md、README_v106.md、README_v103.md。训练检查点与大型求解分片不入包（脚本+种子可再生，索取可得）。

本包为本地同步版本；公开版本化归档（GitHub + DOI）以上传为准，论文可复现性声明以实际归档状态为准。
