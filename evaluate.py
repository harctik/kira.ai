"""
evaluate.py - Free evaluation pipeline: no LLM judge, no API.

Measures retrieval (hit@k, MRR) and answer quality (contains-expected, token F1)
over a golden set: a JSON list of {"question", "expected", "source" (optional)}.

Usage: python evaluate.py golden.json [--index memory/index.json] [--k 3] [--min-hit 0.0]
Exits non-zero when hit@k falls below --min-hit, so it can gate CI.
"""

import argparse
import json
import re
import sys
from collections import Counter
from typing import Dict, List

from retrieval import DocumentIndex, tokenize


def token_f1(pred: str, gold: str) -> float:
    p, g = tokenize(pred), tokenize(gold)
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def contains_expected(pred: str, expected: str) -> bool:
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()  # noqa: E731
    return norm(expected) in norm(pred)


def evaluate_retrieval(index: DocumentIndex, golden: List[Dict], k: int = 3) -> Dict:
    """A hit is a top-k chunk containing the expected text (or, if given, coming from `source`)."""
    hits, rr = 0, 0.0
    for case in golden:
        results = index.search(case["question"], k=k)
        rank = None
        for i, r in enumerate(results, 1):
            ok = contains_expected(r["text"], case["expected"])
            if case.get("source"):
                ok = ok or case["source"] in r["source"]
            if ok:
                rank = i
                break
        if rank:
            hits += 1
            rr += 1 / rank
    n = len(golden) or 1
    return {"n": len(golden), f"hit@{k}": hits / n, "mrr": rr / n}


def evaluate_answers(assistant, golden: List[Dict]) -> Dict:
    contained, f1 = 0, 0.0
    for case in golden:
        answer = assistant.ask(case["question"])["answer"]
        contained += contains_expected(answer, case["expected"])
        f1 += token_f1(answer, case["expected"])
    n = len(golden) or 1
    return {"n": len(golden), "contains_expected": contained / n, "token_f1": f1 / n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("--index", default="memory/index.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-hit", type=float, default=0.0)
    args = ap.parse_args(argv)

    with open(args.golden, "r", encoding="utf-8") as f:
        golden = json.load(f)
    index = DocumentIndex(args.index)
    report = evaluate_retrieval(index, golden, k=args.k)
    print(json.dumps(report, indent=2))
    return 0 if report[f"hit@{args.k}"] >= args.min_hit else 1


if __name__ == "__main__":
    sys.exit(main())
