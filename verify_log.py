"""
解析训练日志，提取 BPB 与 artifact 体积，并检查是否 <= 16MB。
用法:
    python3 verify_log.py logs/JiaJingwen_baseline_8xh100.txt
退出码: 0=通过, 1=超预算/缺字段
"""
import re
import sys

CAP = 16_000_000


def main():
    if len(sys.argv) < 2:
        print("用法: python3 verify_log.py <train_log.txt>")
        sys.exit(2)
    path = sys.argv[1]
    try:
        text = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        print(f"找不到日志: {path}")
        sys.exit(2)

    m_bpb = re.search(r"final_int8_zlib_roundtrip_exact val_bpb:([\d.]+)", text)
    m_size = re.search(r"Total submission size int8\+zlib: (\d+) bytes", text)
    m_loss = re.search(r"final_int8_zlib_roundtrip val_loss:([\d.]+)", text)

    ok = True
    if m_bpb:
        print(f"val_bpb = {float(m_bpb.group(1)):.4f}")
    else:
        print("⚠️ 未找到 val_bpb（训练可能未完成）")
        ok = False
    if m_loss:
        print(f"val_loss = {float(m_loss.group(1)):.4f}")
    if m_size:
        size = int(m_size.group(1))
        print(f"artifact = {size:,} bytes ({size/1e6:.2f} MB)")
        if size > CAP:
            print(f"❌ 超过 16MB 上限 ({CAP:,})！提交会被拒。")
            ok = False
        else:
            print(f"✅ 在 16MB 预算内（剩余 {CAP-size:,} 字节）。")
    else:
        print("⚠️ 未找到 submission size")
        ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
