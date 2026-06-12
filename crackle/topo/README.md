# crackle.topo — TDA branch

把损伤场电影变成:持续同调摘要曲线 + 离散拓扑事件流(h0/h1 born/died),
喂给生存分析/hazard 框架。规格:`crackle_tda_spec.md` v1.0 + addendum
v1.1(交接包 crackle-topo-v2)。

## 模块

| 模块 | 内容 |
|---|---|
| cubical.py | superlevel cubical persistence(cripser/tcripser,T-construction)|
| features.py | 逐帧摘要(Betti、persistence entropy、total persistence)|
| roi.py | 图层面的边界 ROI 过滤(Phase 1.1;场本身不动)|
| matching.py | greedy(Phase 0)与 wasserstein = Hungarian 最优指派(Phase 1.2 默认)|
| events.py | 帧间匹配 → 事件流;`extract_events(..., roi=, method=)` |
| catalog.py | 事件目录(全 provenance)+ (case, step, 6×6-tile) risk sets(Phase 1.3)|
| causal_onset.py | 因果检测器:rolling z / CUSUM + lead/误报评估(Phase 2.1)|
| instability.py | 宏观失稳步 t* 与回溯 lead 表(Phase 0 审计用)|
| ntpp.py | Track A:离散时间 transformer-Hawkes 标记点过程 + 参数 Hawkes 裁判 |
| perslay.py | Track B:scalar / 固定 persistence image / PersLay 学习向量化 |
| synth.py | 多缺口随机运动学世界(数据生成器,非力学求解器)|
| io.py | hetero_pinning npz → (T, ny, nx) 电影 |

脚本:`scripts/phase0_topo_audit.py`(审计)、`generate_topo_dataset.py`
(并行生成)、`build_topo_catalog.py`(目录+risk sets)、
`topo_causal_onset.py`、`topo_hazard_ablation.py`、`topo_track_a_ntpp.py`、
`topo_track_b_perslay.py`、`fetch_datasets.py`。

依赖:`pip install cripser persim scipy scikit-learn pandas pyarrow
xgboost matplotlib`;Track A/B 另需 torch(cu128)。TDA 全程 CPU。

## 状态(2026-06-11,详见 reports/topo_*）

- Phase 0:复验通过(12-case 矩阵逐数字复现)。
- Phase 1:ROI 后边界伪事件 → 0;greedy/wasserstein 分歧 ~0.3%;
  2000-case 目录 kill rule 1 PASS(0.407 ev/step)。
- Phase 2.1:选择性交叉——加速度选择性设置下拓扑信号在 ≥93% case
  报警、中位 lead 23–37 步,total_damage 对照 ≤20%(预注册 2/3 达标);
  宽松设置下对照的"先报警"是斜坡跟随(97.6% 紧贴 growth start)。
- Phase 2.2 表格消融:预注册 PASS——(c)>(b)>(a) 全 horizon,OOD 保持。
- Track A:预注册 PASS——THP 胜参数 Hawkes(test 全 4 项 LL);
  OOD 下 count 项退化(诚实警示)。KS 检验在多数 case 拒绝两个模型。
- Phase 3:三个 API 数据源内容检查全部不符(详见 phase3 报告);
  Rimkus DiB 补充材料为人工获取优先级 1。

## Claim boundary(合成世界)

允许:异质多缺口运动学世界中,拓扑事件流密度充足、分辨率稳健
(48×29 vs 96×58 同世界事件数均值一致,逐 case r=0.76);拓扑特征
在 hazard 预测与事件流建模上超过非拓扑/参数基线(预注册标准)。

不允许:任何真实数据迁移声明(Phase 3 未完成);任何"绝对预测能力
可用"声明(top-1% recall 5–9%);力学定量结论(运动学代理)。
