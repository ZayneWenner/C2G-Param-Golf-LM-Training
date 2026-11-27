# C2G 榜单对照（track_10min_16mb）

**作者**：贾静文　**日期**：2026-11-27

> **数据来源分级**：本表每一行都标注了数字出处。"官方/公开记录"列的数字来自挑战包内 `records/track_10min_16mb/` 各目录的 `submission.json`（可自行打开核验）；"本方案"行标注为**未实测**，不参与排名。

---

## 1. 官方锚点（可核验）

| 方案 | val_bpb | artifact bytes | 出处 |
|---|---|---|---|
| Naive Baseline（SP1024, 9L512d, KV4） | **1.2243657** | 15,863,489 | `2026-03-17_NaiveBaseline/submission.json` |
| 极限约束上限 | — | 16,000,000 | `CHALLENGE.md` |
| 剩余可用空间 | — | **136,511 B**（≈133 KB） | 计算值：16,000,000 − 15,863,489 |

**这两行是本报告的地基。** 它说明：官方 baseline 已经用掉 99.15% 的字节预算，任何净增参数的改动空间只有 133 KB。这是"必须砍层"这一决策的算术来源，也是本方案最容易被复现者忽略的一点。

## 2. 公开历史记录（按目录名自报值，未逐一复算）

| 日期 | 方案（目录名） | 自报 BPB |
|---|---|---|
| 2026-03-18 | `FP16Embed_WD3600` | 见目录内 submission.json |
| 2026-03-19 | `10L_MixedPrecision` | 见目录内 submission.json |
| 2026-03-19 | `int6_STE QAT` | 见目录内 submission.json |
| 2026-03-20 | `11L_XSA4_EMA_Int6_MLP3x_WD04` | 1.1271 |
| 2026-03-21 | `11L_XSA4_EMA_PartialRoPE_LateQAT` | 1.1248 |
| 2026-03-22 | `11L_EMA_GPTQ-lite_warmdown3500_QAT015` | **1.12278022**（已读 submission.json 确认） |
| 2026-04-05 | `SP8192_GPTQ-Embeddings_SDClip_Loop45x2` | 见目录内 submission.json |
| 2026-04-06 | `SP8192_QK5_LegalTTT` | **1.08279384**（已读 submission.json 确认，3 seed 均值） |
| 2026-04-09 | `SP8192_3LayerRecur_ParResid_QK525_LegalTTT` | 见目录内 submission.json |

**只对两个数字声称"已核验"**：1.12278022 与 1.08279384，因为我实际打开了这两个 `submission.json` 并读到了 `val_bpb` 字段。其余数字来自目录名，是提交者自报，**我未逐一复算，因此不标为已核验**。

关于榜首 1.08279384 的构成，其 `submission.json` 写明：11L/512d/8H/4KV、MLP 4x、Partial RoPE 16d、Depth-recurrence、XSA last 4、GPTQ int6、Muon、EMA、Brotli+byte-shuffle，**外加 legal score-first TTT（LR=0.005、3 epochs、freeze 0）**，且 TTT 单独耗时 289–295 秒。这意味着该成绩不是一个"训练 10 分钟"的纯训练结果，而是**训练 + 测试时额外适配**的复合结果。这一点对本方案的路线选择有决定性影响（见 `_ablation.md` §3）。

## 3. 本方案位置（未实测，不参与排名）

| 项目 | 值 | 状态 |
|---|---|---|
| 配置 | SP8192 + 7L512d | 已定 |
| 目标 | < 1.16（落在 Level 2 门槛 1.18 内并留余量） | 未达标验证 |
| 预测区间 | 1.19 – 1.21 | **未实测，仅推断** |
| 实测 val_bpb | null | 待 8×H100 |
| artifact bytes | null | 待实测（估算 ≈15.8 MB） |

**必须直说的结论**：按 `_ablation.md` A3 的净效应推断，本方案预期落在 **1.19–1.21**，**大概率达不到 Level 2（<1.18）**。上表"目标"一栏写 <1.16 是设计意图，不是预测；预测区间才是诚实的判断。若最终实测落在 1.19 附近，我会按 Level 1 报成绩并写明未达 Level 2，而不是挑最好的 seed 汇报。

## 4. 差距归因

从 baseline 1.2244 到榜首 1.0828，差距 0.1416 BPB，拆解如下（基于各记录 submission.json 的 blurb 与 components 字段归纳，**为归因推断而非受控实验**）：

| 来源 | 估计贡献 | 依据 |
|---|---|---|
| 架构与优化器（11L、XSA、Partial RoPE、Muon、bigram hash） | 大 | 3 月记录从 1.2244 走到 1.1228 主要靠这些 |
| 量化与压缩（int6 GPTQ、SD-clip、Brotli、byte-shuffle） | 中 | 让更多参数塞进 16 MB，间接提升容量 |
| 词表 SP8192 | 中 | 4 月记录的共同底座 |
| Legal TTT | 中 | 1.0828 相对其 base PR #1394 增益 0.00283，但整体 TTT 栈贡献更大 |

**这个拆解本身说明：本方案只押了其中两项（词表 + 少量优化器），却期待跨越 0.14 的差距——这不成立。** 因此本方案的目标只能是 Level 1→Level 2 的一步，而非冲榜方案。这一点写在榜单里，比写在别处更有说服力。

## 5. 核验指引

任一数字都可自行复核：

```
records/track_10min_16mb/<目录名>/submission.json
```

查 `val_bpb`（单 seed）或 `mean_bpb`（多 seed 均值）字段。若与上表不一致，以上表为准并请告知我更正——我不希望榜单里留一个抄错的数字。
