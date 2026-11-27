#!/usr/bin/env bash
# =============================================================
# JiaJingwen_C2G · 1 卡小步 smoke 测试（不花钱验证整条管线）
# 目的：在烧钱跑 8 卡前，确认 数据加载→前向→反向→量化→roundtrip 全链路无报错。
# 预期：几分钟内跑完 100 step，日志末尾出现 final_int8_zlib_roundtrip 行即成功。
# =============================================================
set -e
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=0

RUN_ID=JiaJingwen_smoke \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=100 \
TRAIN_BATCH_TOKENS=65536 \
TRAIN_SEQ_LEN=1024 \
MAX_WALLCLOCK_SECONDS=300 \
VAL_LOSS_EVERY=50 \
TRAIN_LOG_EVERY=10 \
torchrun --standalone --nproc_per_node=1 ./JiaJingwen_C2G_train_gpt.py

echo "== 校验日志 =="
python3 verify_log.py "logs/${RUN_ID}.txt"
