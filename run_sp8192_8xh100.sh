#!/usr/bin/env bash
# =============================================================
# JiaJingwen_C2G · Level 2：SP8192 单点改进（8×H100, 10 分钟）
# 目标：val_bpb < 1.18
#
# ⚠️ 16MB 预算硬约束（已用 estimate_params.py 验证）：
#    SP8192 + 9L512d ≈ 20.7M 参数 → int8 后 ~19MB → 【超 16MB，会被拒】
#    所以默认降到 7L512d ≈ 17.1M 参数 → zlib 后 ~15.8MB（刚好卡线，可提交）。
#    若要保留 9 层容量，必须改 int6 (GPTQ) 量化（见 README_运行指南.md 的升级路线）。
#
# 用法：
#   bash run_sp8192_8xh100.sh                 # 默认 7L512d
#   NUM_LAYERS=6 bash run_sp8192_8xh100.sh    # 更保险，约 14.1MB
# =============================================================
set -e
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

export NUM_LAYERS="${NUM_LAYERS:-7}"
export MODEL_DIM="${MODEL_DIM:-512}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MLP_MULT="${MLP_MULT:-2}"
export VOCAB_SIZE=8192

RUN_ID="JiaJingwen_sp8192_${NUM_LAYERS}L${MODEL_DIM}d" \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 ./JiaJingwen_C2G_train_gpt.py

echo "== 校验日志（BPB 与 16MB 体积）=="
python3 verify_log.py "logs/${RUN_ID}.txt"
