#!/usr/bin/env bash
# =============================================================
# JiaJingwen_C2G · Level 1：复现官方 baseline（8×H100, 10 分钟）
# 目标：final_int8_zlib_roundtrip val_bpb ≈ 1.2244 (允许 ±0.005)
# 配置：VOCAB=1024, 9L, dim=512, 8 heads, 4 KV heads, MLP×2, tied embedding
# 参考：records/track_10min_16mb/2026-03-17_NaiveBaseline
# =============================================================
set -e
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

RUN_ID=JiaJingwen_baseline_8xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 ./JiaJingwen_C2G_train_gpt.py

echo "== 校验日志（BPB 与 16MB 体积）=="
python3 verify_log.py "logs/${RUN_ID}.txt"
