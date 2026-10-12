# -*- coding: utf-8 -*-
"""
把 absmom 策略部署到本机 QMT 策略目录（GBK 编码）。

为什么需要这个脚本
------------------
QMT「模型研究」只能导入 `.rzrk`（迅投专有加密二进制，见 {QMT}/template/*.rzrk，
无法在外部生成）；但 QMT 的 Python 策略**源码目录是明文**的：
    {QMT安装目录}/python/*.py
把 .py 直接放进去，QMT 重启后即可在【模型研究】列表中看到并运行，
之后可在 QMT 内右键「导出公式」得到 .rzrk 备份/分享。

用法
----
    python tools/index_lab/deploy_qmt_strategy.py                 # 部署默认策略
    python tools/index_lab/deploy_qmt_strategy.py --exposure 0.6 # 同时切到低回撤档
    python tools/index_lab/deploy_qmt_strategy.py --check        # 只检查环境
"""
import argparse
import os
import shutil
import sys
from datetime import datetime

# 本机 QMT 安装目录（含中文，勿改动）
QMT_HOME = r"D:\中航证券QMT实盘-交易端"
QMT_PY_DIR = os.path.join(QMT_HOME, "python")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "quantaalpha", "strategy", "absmom_qmt_model.py")
SRC = os.path.normpath(SRC)
DST_NAME = "absmom_qmt_model.py"


def _is_plain_python(path):
    """判断文件是否为明文 Python（QMT 保存策略后会加密，此处用于诊断）。"""
    try:
        d = open(path, "rb").read(4096)
    except OSError:
        return None
    return b"def init" in d or b"coding" in d[:80]


def diagnose_qmt_scan():
    """
    诊断 QMT 是否可能识别部署的策略。

    已核实的事实：QMT 保存策略时会**自动加密**写入 python 目录
    （用户新建的策略文件是密文，官方示例才是明文），所以
    「明文 .py 放进去」QMT 未必认 —— 唯一可靠路径是在 QMT 里新建策略并粘贴代码。

    本函数只做**只读**观察，不修改任何文件。
    """
    dst = os.path.join(QMT_PY_DIR, DST_NAME)
    print("-" * 62)
    print("QMT 识别诊断")
    if not os.path.exists(dst):
        print("  [X] 尚未部署: %s" % DST_NAME)
        return
    plain = _is_plain_python(dst)
    if plain:
        print("  [!] 部署版为**明文**。QMT 自身保存的策略是密文,")
        print("      因此 QMT 可能不列出它 -> 走【新建策略+粘贴代码】路线。")
        print("      可粘贴源码: docs/absmom_qmt_paste.txt")
    else:
        print("  [OK] 部署版已是 QMT 认可的形态")

    # 统计目录里明文 vs 密文，帮助判断本QMT 版本的行为
    py = [f for f in os.listdir(QMT_PY_DIR) if f.endswith(".py")]
    enc = [f for f in py if _is_plain_python(os.path.join(QMT_PY_DIR, f)) is False]
    print("  目录内 .py 共 %d 个，其中密文 %d 个（QMT 保存的策略）" % (len(py), len(enc)))
    print("  若重启 QMT 后仍看不到 %s -> 请用粘贴路线" % DST_NAME.replace(".py", ""))


def check_env():
    """检查 QMT 目录与源码是否存在。"""
    ok = True
    print("=" * 62)
    print("QMT 策略目录: %s" % QMT_PY_DIR)
    if not os.path.isdir(QMT_PY_DIR):
        print("  [X] 目录不存在 —— 请改脚本里的 QMT_HOME")
        ok = False
    else:
        n = len([f for f in os.listdir(QMT_PY_DIR) if f.endswith(".py")])
        print("  [OK] 目录存在, 当前 .py 策略 %d 个" % n)

    print("策略源码: %s" % SRC)
    if not os.path.isfile(SRC):
        print("  [X] 源码不存在")
        ok = False
    else:
        print("  [OK] 源码存在 (%d bytes)" % os.path.getsize(SRC))

    print("QMT Python 版本: 3.6.8 (QMT 内置, 语法须兼容 py3.6)")
    diagnose_qmt_scan()
    print("=" * 62)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposure", type=float, default=None,
                    help="改写 EXPOSURE: 1.0=主用档(满仓) 0.6=低回撤档")
    ap.add_argument("--check", action="store_true", help="只检查环境不部署")
    args = ap.parse_args()

    if not check_env():
        if args.check:
            sys.exit(1)
        print("\n环境检查未通过, 已中止。")
        sys.exit(1)
    if args.check:
        return

    # 源码可能已是 GBK（QMT 版），按 gbk 优先读取
    raw = open(SRC, "rb").read()
    try:
        src = raw.decode("utf-8")
    except UnicodeDecodeError:
        src = raw.decode("gbk")

    if args.exposure is not None:
        import re
        src2 = re.sub(r"^EXPOSURE = [\d.]+",
                      "EXPOSURE = %.2f" % args.exposure,
                      src, count=1, flags=re.M)
        if src2 == src:
            print("[X] 未找到 EXPOSURE 行, 请检查源码")
            sys.exit(1)
        src = src2
        print("已将 EXPOSURE 改为 %.2f" % args.exposure)

    # QMT 要求 GBK 编码 + Windows 换行
    payload = src.replace("\r\n", "\n").replace("\n", "\r\n").encode("gbk", errors="replace")
    if not payload.startswith(b"#coding:gbk"):
        print("[X] 首行必须是 #coding:gbk")
        sys.exit(1)

    dst = os.path.join(QMT_PY_DIR, DST_NAME)
    # 备份已有同名文件
    if os.path.exists(dst):
        bak = dst + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(dst, bak)
        print("已备份旧版 -> %s" % os.path.basename(bak))
    with open(dst, "wb") as f:
        f.write(payload)

    # 部署后自检: 能按 gbk 解码 + 能编译
    chk = open(dst, "rb").read().decode("gbk")
    compile(chk, DST_NAME, "exec")

    print("\n部署成功:")
    print("  路径   %s" % dst)
    print("  大小   %d bytes (GBK, CRLF)" % len(payload))
    print("  自检   GBK 解码 OK + 语法编译 OK")
    exp = [l for l in chk.splitlines() if l.startswith("EXPOSURE")]
    if exp:
        print("  档位   %s" % exp[0].strip())
    print("\n下一步（重要）:")
    print("  1) 完全退出 QMT 客户端（含托盘进程），再重新启动")
    print("  2) 【模型研究】列表里应出现 %s" % DST_NAME.replace(".py", ""))
    print("  3) 点【编译】= 保存，然后在【回测】跑 2019-01-01~2026-09-30 对数")
    print("  4) 需要 .rk 备份的话：在 QMT 内右键该策略 → 导出公式")


if __name__ == "__main__":
    main()