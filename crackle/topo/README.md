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
- Phase 2.1 噪声鲁棒性(预注册,有条件 PASS):DIC 式测量噪声下,瞬时
  持久度信号 total_pers_h0 在**相同误报率**下仍胜宏观对照(三个噪声档,
  中位 lead 12 vs 5/11.5/10,检测率高 2–13×,z 探测器)。诚实负结果:
  累积事件计数 cum_events 在噪声下失败;优势依赖探测器。详见
  topo_phase2_onset_noise_20260612.md。
- Phase 2.2 表格消融:预注册 PASS——(c)>(b)>(a) 全 horizon,OOD 保持。
- Track A:预注册 PASS——THP 胜参数 Hawkes(test 全 4 项 LL);
  OOD 下 count 项退化(诚实警示)。KS 检验在多数 case 拒绝两个模型。
- Track B:预注册 FAIL = 负结果——PersLay/固定 PI 不敌手工标量曲线。
- Track C:预注册 PASS 3/3(最强)——键图 GNN 比同特征+一跳聚合的 GBM
  裁判低 35–58% NLL,OOD 下优势反而扩大。
- Phase 2.2 噪声消融(预注册 PASS):σ=0.05 噪声下 GBM (c) 拓扑特征仍
  击败 (a) 纯局部 + 全部传统裁判(test+OOD 全 horizon);拓扑优势间隙
  在噪声下完全保留——打赢传统法不是 clean 侥幸。
- **Phase 3.2 真实数据(DONE,正结果):** Harb RC 板 25 加载阶段裂纹掩码
  (Zenodo 15187675,CC-BY-4.0)。合成事件语法复现:h0_born 主导、核化
  先于成环(中位 epoch 13 vs 19)、H0/H1 单调增长;对降采样/阈值/匹配器
  鲁棒。n=1 试件。脚本 `scripts/topo_realdata_audit.py`,loader
  `load_mask_sequence`。详见 topo_phase3_harb_audit_20260612.md。
- 整合综述:reports/topo_branch_writeup_20260612.md。

## Claim boundary(合成世界)

允许:异质多缺口运动学世界中,拓扑事件流密度充足、分辨率稳健
(48×29 vs 96×58 同世界事件数均值一致,逐 case r=0.76);拓扑特征
在 hazard 预测与事件流建模上超过非拓扑/参数基线(预注册标准)。

不允许:任何真实数据迁移声明(Phase 3 未完成);任何"绝对预测能力
可用"声明(top-1% recall 5–9%);力学定量结论(运动学代理)。
