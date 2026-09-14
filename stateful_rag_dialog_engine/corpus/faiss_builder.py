"""
faiss_builder.py —— 为每个 txt 建 FAISS 索引。

对应原代码:
    save_db.py

与原代码的关键差异:
    1. embeddings 在循环外创建一次，避免 N 个 txt 重复加载模型
       （原代码每个文件都 HuggingFaceEmbeddings(...)，37 个文件加载 37 次）
    2. device 从配置读，不硬编码 cuda
    3. 模型路径从配置读，不硬编码
    4. 遍历已完成的 txt 集合，而不是靠目录 ls
    5. 单个文件失败不影响其他文件，最终汇总报告
    6. 幂等：每个 faiss 目录由 save_local 覆盖，不残留旧数据
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CorpusConfig

logger = logging.getLogger(__name__)


# ==================================================================
# 报告
# ==================================================================

@dataclass
class FaissReport:
    built: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    total_questions: int = 0
    elapsed_seconds: float = 0.0

    @property
    def summary(self) -> str:
        return (
            f"built={len(self.built)}, "
            f"skipped={len(self.skipped)}, "
            f"failed={len(self.failed)}, "
            f"questions={self.total_questions}, "
            f"time={self.elapsed_seconds:.1f}s"
        )


# ==================================================================
# 解析 txt 为 questions
# ==================================================================

def parse_questions_from_txt(text: str) -> list[str]:
    """
    从 txt 内容提取所有 Question 行。

    与原 save_db.py 一致:
        lines = text.split("\\n")
        for i in range(0, len(lines), 3):
            if i+2 < len(lines):
                qa_pairs.append((lines[i], lines[i+1], lines[i+2]))
        questions = [q for q, a, l in qa_pairs]

    每 3 行是一个 QA 块: Question / Answer / Label
    """
    lines = text.split("\n")
    questions: list[str] = []
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            questions.append(lines[i])
    return questions


def build_faiss_document(questions: list[str]) -> Document:
    """
    把 questions 拼成一整个 Document，交给 splitter 再切。

    与原 save_db.py 一致:
        db_saves(''.join(questions), ...)
    """
    return Document(metadata={}, page_content="".join(questions))


# ==================================================================
# 单个 txt -> FAISS
# ==================================================================

def build_one(
    txt_path: Path,
    output_dir: Path,
    embeddings,
    text_splitter: RecursiveCharacterTextSplitter,
) -> int:
    """
    为一个 txt 建 FAISS 并 save_local。

    返回: 该文件包含的 question 数
    """
    # 延迟导入：langchain_community 是重依赖，只在真正跑时加载
    from langchain_community.vectorstores import FAISS

    text = txt_path.read_text(encoding="utf-8")
    questions = parse_questions_from_txt(text)

    if not questions:
        raise ValueError(f"未解析出任何 Question: {txt_path.name}")

    doc = build_faiss_document(questions)
    docs = text_splitter.split_documents([doc])

    db = FAISS.from_documents(docs, embeddings)
    output_dir.mkdir(parents=True, exist_ok=True)
    db.save_local(str(output_dir))

    return len(questions)


# ==================================================================
# 批量
# ==================================================================

def build_all(config: CorpusConfig) -> FaissReport:
    """
    为 config.output.txt_dir 下所有 .txt 建 FAISS。

    输出到 config.output.faiss_dir / {stem}/
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    report = FaissReport()
    start = time.time()

    txt_dir = config.output.txt_dir
    faiss_dir = config.output.faiss_dir

    if not txt_dir.exists():
        raise FileNotFoundError(f"txt_dir 不存在: {txt_dir}")

    # 收集所有 txt 文件
    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        logger.warning(f"{txt_dir} 下没有 .txt 文件")
        return report

    logger.info(f"发现 {len(txt_files)} 个 txt 文件，准备建 FAISS")
    logger.info(
        f"embedding 模型: {config.embedding.model_path} "
        f"(device={config.embedding.device})"
    )

    # ── 关键：embeddings 只创建一次 ──
    t0 = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=str(config.embedding.model_path),
        model_kwargs={"device": config.embedding.device},
    )
    logger.info(f"embeddings 加载完成，耗时 {time.time() - t0:.1f}s")

    # splitter 也复用
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10,
        chunk_overlap=0,
        length_function=len,
        separators=["Question:"],
    )

    faiss_dir.mkdir(parents=True, exist_ok=True)

    for i, txt_path in enumerate(txt_files, 1):
        stem = txt_path.stem
        out_dir = faiss_dir / stem

        try:
            t0 = time.time()
            n_q = build_one(txt_path, out_dir, embeddings, text_splitter)
            report.built.append(out_dir)
            report.total_questions += n_q
            logger.info(
                f"[{i:>2d}/{len(txt_files)}] {stem} "
                f"({n_q} q, {time.time() - t0:.2f}s)"
            )
        except Exception as e:
            logger.error(f"[{i:>2d}/{len(txt_files)}] {stem} 失败: {e}")
            report.failed.append((txt_path, str(e)))

    report.elapsed_seconds = time.time() - start
    logger.info(f"FAISS 构建完成: {report.summary}")
    return report


# ==================================================================
# 自检
# ==================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m stateful_rag_dialog_engine.corpus.faiss_builder <corpus.yaml>")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s | %(message)s",
    )

    from .config import load_config

    cfg = load_config(sys.argv[1])
    report = build_all(cfg)

    print()
    print("=" * 78)
    print("FAISS 构建报告")
    print("=" * 78)
    print(f"  built   : {len(report.built)} 个")
    print(f"  skipped : {len(report.skipped)} 个")
    print(f"  failed  : {len(report.failed)} 个")
    print(f"  questions 总计: {report.total_questions}")
    print(f"  总耗时: {report.elapsed_seconds:.1f}s")
    if report.failed:
        print()
        print("  失败列表:")
        for p, err in report.failed:
            print(f"    - {p.name}: {err}")
    print()
    print(f"  输出目录: {cfg.output.faiss_dir}")