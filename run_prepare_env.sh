#!/usr/bin/env bash
# =============================================================
# JiaJingwen_C2G · 环境准备 + 数据下载
# 在云 GPU 实例（Linux）上运行，本机（无 GPU）不能训练，只能准备。
# =============================================================
set -e
cd "$(dirname "$0")"

echo "== [1/3] 创建 venv 并安装依赖 =="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_c2g.txt

echo "== [2/3] 下载 baseline 数据 (sp1024, 官方默认仓库) =="
# 默认从 willdepueoai/parameter-golf 拉取，含 sp1024 / sp4096
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

echo "== [3/3] 下载 SP8192 数据 + tokenizer (kevclark/parameter-golf) =="
# 注意：SP8192 数据在独立仓库，需先删旧 manifest 再切换 REPO_ID
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128

echo ""
echo "✅ 环境就绪。下一步："
echo "   bash run_smoke.sh              # 1 卡小步验证管线"
echo "   bash run_baseline_8xh100.sh    # 8 卡复现 baseline (目标 1.2244)"
echo "   bash run_sp8192_8xh100.sh      # 8 卡 SP8192 (目标 <1.18)"
echo ""
echo "⚠️ 若之后又要跑 sp1024 baseline，直接重新执行:"
echo "   python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80"
echo "   (它会重新拉取默认仓库的 manifest；.bin 文件已缓存，不会重复下载)"
