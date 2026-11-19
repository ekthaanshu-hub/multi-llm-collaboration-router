#!/usr/bin/env python3
"""Train a small multinomial Naive Bayes question router using only stdlib."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

LABELS = {
    "evidence_extractor": "Single-paper definitions, motivation, or setup questions",
    "method_analyst": "Questions about methods, mechanisms, or reported results",
    "cross_paper_synthesizer": "Questions requiring comparison or synthesis across papers",
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    features = [f"u:{word}" for word in words if len(word) > 1]
    features.extend(f"b:{left}_{right}" for left, right in zip(words, words[1:]))
    return features


def train(rows: list[dict[str, str]], alpha: float = 1.0) -> dict:
    if not rows:
        raise ValueError("Training data is empty")
    class_docs: Counter[str] = Counter()
    class_terms: dict[str, Counter[str]] = defaultdict(Counter)
    vocabulary: set[str] = set()

    for row in rows:
        label = row["route_label"].strip()
        question = row["question"].strip()
        if label not in LABELS:
            raise ValueError(f"Unknown route_label: {label}")
        if not question:
            raise ValueError(f"Empty question for {row.get('item_id', '?')}")
        class_docs[label] += 1
        features = tokenize(question)
        class_terms[label].update(features)
        vocabulary.update(features)

    vocab = sorted(vocabulary)
    vocab_size = len(vocab)
    total_docs = len(rows)
    log_prior = {}
    log_likelihood = {}
    term_total = {}
    for label in LABELS:
        if not class_docs[label]:
            continue
        log_prior[label] = math.log(class_docs[label] / total_docs)
        total = sum(class_terms[label].values())
        term_total[label] = total
        denom = total + alpha * (vocab_size + 1)
        log_likelihood[label] = {
            term: math.log((class_terms[label][term] + alpha) / denom)
            for term in vocab
        }
        log_likelihood[label]["<UNK>"] = math.log(alpha / denom)

    return {
        "model_name": "question-role-router",
        "algorithm": "multinomial_naive_bayes_unigram_bigram",
        "model_version": 1,
        "training_rows": total_docs,
        "alpha": alpha,
        "classes": {label: LABELS[label] for label in log_prior},
        "class_counts": {label: class_docs[label] for label in log_prior},
        "vocabulary": vocab,
        "class_log_prior": log_prior,
        "token_log_likelihood": log_likelihood,
        "class_feature_totals": term_total,
        "evaluation_note": "No held-out evaluation was performed; ten examples are insufficient for a generalization claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.data.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    required = {"item_id", "question", "route_label"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"CSV must contain columns: {', '.join(sorted(required))}")
    model = train(rows)
    model["training_data_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Trained {model['algorithm']} on {model['training_rows']} questions.")
    print(f"Classes: {model['class_counts']}")
    print(f"Model saved to {args.output}")
    print("No held-out accuracy is reported; see README for the sample-size limitation.")


if __name__ == "__main__":
    main()
