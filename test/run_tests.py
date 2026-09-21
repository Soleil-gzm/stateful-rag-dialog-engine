"""
对话引擎的 golden test。

用法：
    # 第一次用（或在确认新行为正确后，更新基准）
    python test/run_tests.py --update

    # 平时跑回归测试
    python test/run_tests.py

    # 想同时看 stderr（调试用）
    python test/run_tests.py --verbose
"""
import argparse
import difflib
import subprocess
import sys
from pathlib import Path

# ---- 路径配置 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "test"
INPUT_FILE = TEST_DIR / "test_input.jsonl"
GOLDEN_FILE = TEST_DIR / "golden.txt"
ACTUAL_FILE = TEST_DIR / "actual.txt"     # 每次跑的临时输出
ARGS_FILE = PROJECT_ROOT / "config" / "args" / "qwen.json"


def run_engine() -> tuple[str, str]:
    """跑一次对话引擎，返回 (stdout, stderr)。"""
    if not INPUT_FILE.exists():
        sys.exit(f"❌ 找不到输入文件: {INPUT_FILE}")

    cmd = [
        sys.executable, "-m", "stateful_rag_dialog_engine",
        "--args_file", str(ARGS_FILE),
    ]
    print(f"执行: {' '.join(cmd)}")
    print(f"输入: {INPUT_FILE}")

    with open(INPUT_FILE, "rb") as fin:
        proc = subprocess.run(
            cmd,
            stdin=fin,
            capture_output=True,
            cwd=PROJECT_ROOT,      # 保证相对路径找得到
        )

    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        print(f"\n⚠️ 引擎退出码 = {proc.returncode}")
        print("---- stderr 尾部 ----")
        print("\n".join(stderr.splitlines()[-30:]))
        sys.exit(1)

    return stdout, stderr


def update_golden(stdout: str):
    """把当前输出存成新的 golden。"""
    GOLDEN_FILE.write_text(stdout, encoding="utf-8")
    print(f"\n✅ 已更新基准: {GOLDEN_FILE}")
    print(f"   共 {len(stdout.splitlines())} 行")


def compare(stdout: str) -> bool:
    """对比当前输出和 golden，返回是否一致。"""
    if not GOLDEN_FILE.exists():
        print(f"\n❌ 找不到基准文件: {GOLDEN_FILE}")
        print("   第一次请先跑: python test/run_tests.py --update")
        return False

    golden = GOLDEN_FILE.read_text(encoding="utf-8")

    # 保存当前输出，方便出问题后自己看
    ACTUAL_FILE.write_text(stdout, encoding="utf-8")

    if stdout == golden:
        n = len(stdout.splitlines())
        print(f"\n✅ 通过：输出与基准一致（{n} 行）")
        return True

    # 不一致：打印 diff
    print("\n❌ 失败：输出与基准不一致\n")
    diff = difflib.unified_diff(
        golden.splitlines(keepends=False),
        stdout.splitlines(keepends=False),
        fromfile="golden.txt",
        tofile="actual.txt",
        lineterm="",
        n=1,     # 每处差异前后显示 1 行上下文
    )
    for line in diff:
        if line.startswith("+"):
            print(f"\033[32m{line}\033[0m")   # 绿色 = 新增
        elif line.startswith("-"):
            print(f"\033[31m{line}\033[0m")   # 红色 = 删除
        else:
            print(line)

    print(f"\n完整输出已保存到: {ACTUAL_FILE}")
    print("查看命令: diff test/golden.txt test/actual.txt")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="更新 golden 基准（仅在确认新行为正确时用）")
    parser.add_argument("--verbose", action="store_true",
                        help="同时打印 stderr")
    args = parser.parse_args()

    stdout, stderr = run_engine()

    if args.verbose:
        print("\n---- stderr ----")
        print(stderr)

    if args.update:
        update_golden(stdout)
        sys.exit(0)

    ok = compare(stdout)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()