# -*- coding: utf-8 -*-
"""
生成「可直接粘贴进 QMT」的策略代码文件。

为什么是「粘贴」而不是「自动部署」
--------------------------------
QMT【模型研究】只能导入 `.rzrk`（迅投专有）。实测：
  - `.rzrk` 是强加密二进制（熵 7.94~7.97，8.0=完全随机），外部无法生成；
  - QMT **保存策略时会自动加密**写入 `{QMT}/python/*.py`
    （用户新建的策略是密文；官方示例 `PY模型回测示例.py` 才是明文），
    所以把明文 .py 放进该目录，QMT 不认。

结论：唯一可行路径是**在 QMT 里新建 Python 策略，把代码粘贴进去，点【编译】**，
由 QMT 自己加密落盘并生成 .rzrk。本脚本只负责产出这份可粘贴的代码文件。

用法
----
    python tools/index_lab/make_paste_code.py# 生成 docs/absmom_qmt_paste.txt
    python tools/index_lab/make_paste_code.py --check        # 只校验，不写文件
    python tools/index_lab/make_paste_code.py --exposure 0.6 # 顺便切到低回撤档

产物
----
    docs/absmom_qmt_paste.txt   UTF-8 + BOM + LF（编辑器打开不乱码）
"""
import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..",
                                    "quantaalpha", "strategy",
                                    "absmom_qmt_model.py"))
OUT_PROJECT = os.path.normpath(os.path.join(HERE, "..", "..",
                                            "docs", "absmom_qmt_paste.txt"))
QMT_HOME = r"D:\中航证券QMT实盘-交易端"
OUT_QMT = os.path.join(QMT_HOME, "absmom_策略代码_可直接粘贴.txt")


def read_source():
    """读取策略源码。源可能是 GBK（QMT 版要求），按 GBK 优先解码。"""
    raw = open(SRC, "rb").read()
    for enc in ("gbk", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("源码既不是 GBK 也不是 UTF-8，请检查 %s" % SRC)


def set_exposure(text, exposure):
    """改写 EXPOSURE 的值，返回新文本；找不到则抛错。"""
    new, n = re.subn(r"^EXPOSURE = [\d.]+",
                     "EXPOSURE = %.2f" % exposure,
                     text, count=1, flags=re.M)
    if n != 1:
        raise ValueError("源码中未找到 EXPOSURE 行，请检查 %s" % SRC)
    return new


def check(text):
    """静态校验：语法、编码可转性、关键 API 存在。返回警告列表。"""
    warnings = []
    compile(text, "absmom", "exec")            # 语法
    if not text.startswith("#coding:gbk"):
        warnings.append("首行不是 #coding:gbk，QMT 解析可能失败")

    # ★编码声明必须唯一且只占第1行。Python 只认第1行的 coding 声明，
    #   若第2行再写一次 utf-8 而实际内容是 GBK（或反之），
    #   编译时会报 "codec can't decode byte 0x.." ��这类极难定位的错。
    lines = text.split("\n")
    decl = [(i + 1, l.strip()) for i, l in enumerate(lines[:5])
            if re.search(r"coding[:=]\s*[\w-]+", l)]
    if len(decl) > 1:
        warnings.append("第1-5行有多个编码声明 %s，Python只认第1行 -> 必须只保留第1行"
                        % [d[1] for d in decl])
    if decl and decl[0][0] != 1:
        warnings.append("编码声明出现在第 %d 行，必须在第 1 行" % decl[0][0])

    # 落地字节自检：整段文本必须能原样编码成 GBK（QMT 的实际解码方式）
    try:
        text.encode("gbk")
    except UnicodeEncodeError as e:
        warnings.append("存在无法转 GBK 的字符: %s" % e)
    if "import pandas" in text:
        warnings.append("依赖 pandas，QMT 自带但建议只留 numpy")
    for api in ("passorder", "get_trade_detail_data", "is_last_bar",
                "get_market_data_ex"):
        if api not in text:
            warnings.append("缺少关键 QMT API: %s" % api)
    return warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验不写文件")
    ap.add_argument("--exposure", type=float, default=None,
                    help="改写 EXPOSURE: 1.0=主用档 0.6=低回撤档")
    ap.add_argument("--no-qmt-copy", action="store_true",
                    help="不复制到 QMT 目录（默认会复制一份便于取用）")
    args = ap.parse_args()

    print("策略源码: %s" % SRC)
    if not os.path.isfile(SRC):
        print("[X] 源码不存在")
        sys.exit(1)

    text = read_source()
    if args.exposure is not None:
        text = set_exposure(text, args.exposure)
        print("已将 EXPOSURE 改为 %.2f" % args.exposure)

    warns = check(text)
    print("校验: 语法通过 / 可转 GBK")
    for w in warns:
        print("  [!] %s" % w)

    exp = [l for l in text.splitlines() if l.startswith("EXPOSURE")]
    print("  档位 %s" % (exp[0].strip() if exp else "未找到 EXPOSURE"))
    print("  行数 %d" % (text.count("\n") + 1))

    if args.check:
        return
    if warns:
        print("\n[X] 存在警告，未生成。请先修正。")
        sys.exit(1)

    # 两份文件用不同编码，各有其道理：
    #  · docs 那份 = 给人看的 -> UTF-8+BOM，编辑器/记事本打开不乱码
    #  · QMT 那份 = 给 QMT 读的 -> **必须 GBK + 无 BOM**。
    #    QMT 按 GBK 处理策略文本，若给 UTF-8 会报
    #    "codec can't decode byte 0x.."（实测踩过）。
    #    代价：用 UTF-8 编辑器打开这份会看到乱码，属正常 —— 只复制内容、别看内容。
    os.makedirs(os.path.dirname(OUT_PROJECT), exist_ok=True)
    with open(OUT_PROJECT, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write(text)
    print("  已生成 %s (%d bytes, UTF-8+BOM / 供查阅)" % (OUT_PROJECT, os.path.getsize(OUT_PROJECT)))

    if not args.no_qmt_copy:
        os.makedirs(os.path.dirname(OUT_QMT), exist_ok=True)
        with open(OUT_QMT, "wb") as f:
            f.write(text.replace("\r\n", "\n").encode("gbk"))
        # 自检：按 QMT 的方式读回来并编译
        chk = open(OUT_QMT, "rb").read()
        assert chk[:3] != b"\xef\xbb\xbf", "QMT 版不应带 BOM"
        compile(chk.decode("gbk"), "absmom", "exec")
        print("  已生成 %s (%d bytes, GBK / 给QMT粘贴, 已自检通过)"
              % (OUT_QMT, os.path.getsize(OUT_QMT)))

    print("\n下一步（QMT 内操作）:")
    print("  1)【我的】→ 新建策略 → Python 策略")
    print("  2) 打开上面任一文件，全选复制，粘贴进编辑器")
    print("  3) 点【编译】→ 关闭，策略即在列表中")
    print("  4) 点【回测】(不是【模型交易】) 区间 2019-01-01~2026-09-30 初始资金 80000")
    print("  5) 需要 .rzrk 备份：右键策略 →「导出公式」")


if __name__ == "__main__":
    main()