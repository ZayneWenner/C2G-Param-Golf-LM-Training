# C2G 方案设计（v2）

**作者**：贾静文　**日期**：2026-11-27　**赛道**：`track_10min_16mb`

> **证据边界（先读这条）**：本方案的全部结论属**设计推理 + 文献/官方记录引用**，**均未实测**——本机无 GPU、无 PyTorch、无 CUDA（详见 §6）。文中 `1.2243657`、`1.08279384` 等数字来自挑战包 `records/track_10min_16mb/` 内各 `submission.json`，属**引用**而非本人产出；本方案自身的 `val_bpb` 在 `_submission.json` 中留 `null`，待 8×H100 实跑后回填。
>
> v2 修订说明：v1（2026-10-07）把目标定在 Level 3（<1.12）。经复核官方 `records/track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828/submission.json` 后确认，1.08 级别依赖 **legal score-first TTT（训练后测试时训练）** 这一整套额外机制，而不只是词表与层数。在只有约 6 次 H100 run 的预算下追 1.12 是不诚实的，**v2 把目标下调为 Level 2（< 1.18）**，并把 TTT 明确列为非目标。

---

## 1. 目标与非目标

**目标（可验收）**
- G1：在 `8×H100 SXM` / 600 秒 / 16,000,000 字节三项硬约束下，跑满 3 个 seed（1 / 42 / 1337），records 完整。
- G2：`val_bpb` 三 seed 均值 **< 1.18**（Level 2），相对官方 baseline 1.2243657 至少改善 0.044。
- G3：任意单个 seed 的 `artifact_bytes ≤ 16,000,000`、`train_time ≤ 600 s`，由脚本断言而非赛后手算。
- G4：另一个工程师拿到本仓库，不改一行代码即可复跑出同一量级的 BPB。

**非目标（明确不做）**
- 不追 Level 3/4：不引入 TTT、不做 SLOT/ETLB、不做 n-gram cache。这些机制在 `CHALLENGE.md` 的合规条款里已被单独约束，且在 6 次 run 的预算内无法同时调稳。
- 不做多方向并行的"参数扫描"：见 §2 的单一主轴原则。
- 不在本地声称任何 BPB：本机无 GPU（见 §6）。

## 2. 技术方案

### 2.1 单一主轴：SP8192 词表

`CHALLENGE.md` 的 Step 3 把 SP8192 列为首选，历史增益 **−0.03 ~ −0.05 BPB**，理由是"改动集中在预处理，不动模型"。选择它还有一个工程理由：`train_gpt.py` 全部由环境变量驱动（`DATA_PATH`、`TOKENIZER_PATH`、`VOCAB_SIZE`、`NUM_LAYERS`、`MODEL_DIM` 等 40 余项），换词表**不需要改代码**，只需要改环境变量与数据集路径。在只有 6 次 run 的预算里，"零代码改动"本身就是最高的性价比。

### 2.2 参数预算的硬算术

这是本方案最关键的一步，也是多数人踩坑的地方。

| 配置 | 参数量 | int8+zlib 估算 artifact |
|---|---|---|
| SP1024 + 9L512d（官方 baseline，实测） | 17.06 M | **15,863,489 B** |
| SP8192 + 9L512d | 20.73 M | ≈ 19.2 MB ❌ 超限 |
| SP8192 + 7L512d（**本方案**） | 17.06 M | ≈ 15.8 MB ✔ |
| SP8192 + 6L512d（兜底） | 15.22 M | ≈ 14.1 MB ✔ |

词表 1024→8192 使 embedding/head 从 0.52 M 涨到 4.19 M 参数，按 int8 约 +3.7 MB。官方 baseline 实测只剩 0.14 MB 余量，因此**必须同步砍层**。每砍一层 512d/mlp2x 约省 1.9 M 参数（≈1.9 MB int8）。9→7 正好抵消词表增量并留出约 0.2 MB 余量。

> **所以"SP8192 + 7L"是一个不可拆分的单一决策**，不是两个可独立开关的旋钮。把这条写清楚，是为了防止复现者只改词表就撞上 16 MB 上限。

### 2.3 关键环境变量（Route A）

```bash
export DATA_PATH=./data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export VOCAB_SIZE=8192
export NUM_LAYERS=7
export MODEL_DIM=512
export MAX_WALLCLOCK_SECONDS=600
export SEED=1
torchrun --nproc_per_node=8 JiaJingwen_C2G_train_gpt.py
```

**只设 `VOCAB_SIZE` 是不够的。** `train_gpt.py` 的两个默认路径仍指向 `fineweb10B_sp1024` 与 `fineweb_1024_bpe.model`，必须同时覆盖 `DATA_PATH` 与 `TOKENIZER_PATH`，否则会出现"词表声明是 8192、数据却是 1024 编码"的静默错训——loss 照降，成绩全废。这是本方案的第一号风险，`verify_log.py` 负责拦截。

## 3. 实验矩阵

| ID | 配置 | seed | 目的 | 状态 |
|---|---|---|---|---|
| E000 | SP1024 + 9L512d | 1337 | 复现 1.2244，验证环境 | 待跑 |
| E001 | SP8192 + 7L512d | 1 | 主结果 | 待跑 |
| E002 | SP8192 + 7L512d | 42 | 主结果 | 待跑 |
| E003 | SP8192 + 7L512d | 1337 | 主结果 | 待跑 |
| E004 | SP8192 + 6L512d | 1337 | 兜底：若 E001 超 16 MB | 待跑 |
| E005 | SP8192 + 7L512d + `QK_GAIN_INIT=5` | 1337 | 可选：优化器侧增益 | 待跑 |

**预算对齐**：E000–E003 为必跑 4 次（40 min），E004/E005 为可选 2 次（20 min），加 20 min 环境重装 buffer，合计 ≈ 80 min，对应 $25 档位。

## 4. 验收标准

| 项 | 判据 | 在何处证据化 |
|---|---|---|
| 成绩 | 3 seed 均值 `val_bpb < 1.18` | `JiaJingwen_C2G_leaderboard.md` |
| 合规 | `artifact_bytes ≤ 16,000,000` 且 `train_time ≤ 600 s` | 每次 run 的 `submission.json` |
| 方差 | 3 seed 极差 < 0.005，否则视为未收敛 | `_ablation.md` |
| 可复现 | 他人按 §2.3 命令走出同量级结果 | `_拿来说明.md` + `_logs/` |

**任一不达标即如实写入报告，不修饰。**

## 5. 执行清单（拿到 H100 后按序）

1. `bash run_prepare_env.sh` — 装 torch 2.11.0+cu130、下载 `fineweb10B_sp8192` 与 8192 BPE 模型
2. `bash run_smoke.sh` — 单卡小步数冒烟，跑 `verify_log.py` 抽查解码文本（**这一步必须人眼看**）
3. `bash run_baseline_8xh100.sh` — E000
4. `bash run_sp8192_8xh100.sh` — E001/E002/E003，逐 seed 改 `SEED`
5. `python verify_log.py` + `python JiaJingwen_C2G_pack.py` — 生成 artifact 并断言尺寸
6. 把 3 次 run 的 `submission.json` 回填到 `JiaJingwen_C2G_submission.json`

## 6. 风险与当前状态

| 风险 | 影响 | 缓解 |
|---|---|---|
| tokenizer 与数据导出不匹配 | 静默错训，成绩全废 | `verify_log.py` 解码抽查 + baseline 复现校验 |
| 7L 容量不足吃掉词表收益 | 达不到 <1.18 | E004（6L）对照，把权衡曲线写进消融表 |
| 8×H100 显存/通信异常 | run 失败 | 预留 20 min buffer；`NCCL socket` 已记录 |
| 无本地 GPU | **无法本地实测** | 全部实测留待云端，本文档不含任何伪造数字 |

**当前真实状态（2026-11-27）**：本机 Python 3.12.13、无 PyTorch、无 `nvidia-smi`、无可用 GPU。因此本方案**已具备可执行性，但尚未产出任何实测 BPB**。`_logs/` 为空目录并附说明；`submission.json` 的成绩字段为 `null`。这是刻意保留的诚实缺口，不是遗漏。
