# JiaJingwen_C2G_logs —— 运行日志目录

**作者**：贾静文 **日期**：2026-11-27

---

## 当前状态：本目录暂无本人训练日志

**原因**：本机环境不具备 C2G 运行条件——Python 3.12.13 已装，但**无 PyTorch、无 CUDA、无 `nvidia-smi`**。C2G 的最低运行门槛是 8×H100 SXM（10 分钟 wallclock），消费级环境无法替代。

**因此**：本提交包中**不存在任何本人实测的 `val_bpb`**。`JiaJingwen_C2G_submission.json` 的 `name` 字段已标注 `pending measurement`，`val_bpb` 等数值字段留 `null`，`seed_results` 为空数组。这是刻意的——不用文献值填补自己的空位。

---

## 这个目录要装什么（回填规范）

跑通后，每次正式 run 在 `JiaJingwen_C2G_logs/` 下建一个子目录，命名：

```
JiaJingwen_C2G_logs/seed_<SEED>/
```

每个子目录内必须落盘两类文件：

| 文件 | 内容 | 来源 |
|---|---|---|
| `train.log` | 完整 stdout/stderr | `run_sp8192_8xh100.sh` 重定向 |
| `submission.json` | 该次 run 的结构化结果 | `train_gpt.py` 训练结束时自动写出 |

`submission.json` 必须包含的字段（与官方记录格式一致）：

```json
{
  "val_loss": <float>,
  "val_bpb": <float>,
  "artifact_bytes": <int>,
  "steps": <int>
}
```

三条正式 run 对应 `seed_1 / seed_42 / seed_1337`（计划 seeds `[1, 42, 1337]`）。

---

## 日志核验方法

包内已附 `verify_log.py`。用法：

```bash
python verify_log.py JiaJingwen_C2G_logs/seed_1/train.log
```

核验通过的最低标准：
1. 日志中出现 `VOCAB_SIZE=8192`（证明 tokenizer 覆盖生效，不是默认 1024 跑出来的）；
2. 出现 `NUM_LAYERS=7`、`MODEL_DIM=512`；
3. 末尾有 `val_bpb` 数值；
4. 训练时长 ≤ 600 s；
5. 写出的 `artifact_bytes` ≤ 16,000,000。

---

## 复现脚本清单（均在包根目录）

| 脚本 | 作用 | 何时用 |
|---|---|---|
| `run_prepare_env.sh` | 装依赖、拉 tokenizer 与数据集 | 第一步，必跑 |
| `run_smoke.sh` | 冒烟测试：只验能启动、能出 BPB，不追分数 | 第二步，上 8 卡前必跑 |
| `run_baseline_8xh100.sh` | 官方 baseline（SP1024 / 9L512d），校验 `≈1.2244` | 环境自检 |
| `run_sp8192_8xh100.sh` | 路线 A（SP8192 / 7L512d）三次 seed | 正式运行 |
| `verify_log.py` | 日志字段核验 | 每次 run 后 |

---

## 引用说明

`_leaderboard.md` 与 `_ablation.md` 中出现的官方数字（如 `1.2243657`、`1.08279384`）来自挑战包内 `records/track_10min_16mb/` 各目录的 `submission.json`，属**引用**，可自行打开核验，**不是本人在本目录产出的日志**。两者的区别在本目录中不做模糊处理。
