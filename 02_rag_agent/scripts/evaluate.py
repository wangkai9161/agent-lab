from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import load_settings
from rag.rag_agent import ExperimentRagAgent


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval behavior")
    parser.add_argument("--eval-file", default="data/eval/questions.jsonl")
    parser.add_argument("--docs", default=str(settings.docs_dir))
    parser.add_argument("--index-path", default=str(settings.index_path))
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--min-confidence", type=float, default=settings.min_confidence)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, object]]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs)
    index_path = Path(args.index_path)
    agent = ExperimentRagAgent(min_confidence=args.min_confidence)

    if index_path.exists() and not args.rebuild_index:
        agent.load_index(index_path)
    else:
        agent.ingest_directory(docs_dir)
        agent.save_index(index_path)

    cases = load_cases(Path(args.eval_file))
    passed = 0

    for case in cases:
        question = str(case["question"])
        expected_sources = set(case.get("expected_sources", []))
        should_refuse = bool(case.get("should_refuse", False))
        response = agent.answer_with_sources(question, top_k=args.top_k)
        actual_sources = {source["file_name"] for source in response.sources}
        source_ok = expected_sources.issubset(actual_sources)
        refuse_ok = response.refused is should_refuse
        ok = source_ok and refuse_ok
        passed += int(ok)

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {question}")
        print(f"  expected_sources={sorted(expected_sources)}")
        print(f"  actual_sources={sorted(actual_sources)}")
        print(f"  refused={response.refused}, expected_refused={should_refuse}")

    total = len(cases)
    print(f"\nSummary: {passed}/{total} passed")


if __name__ == "__main__":
    main()
