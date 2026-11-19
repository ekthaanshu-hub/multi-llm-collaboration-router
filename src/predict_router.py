#!/usr/bin/env python3
"""Predict a suggested workflow role using a saved question-router model."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    features = [f"u:{word}" for word in words if len(word) > 1]
    features.extend(f"b:{left}_{right}" for left, right in zip(words, words[1:]))
    return features


def predict(question: str, model: dict) -> dict:
    vocab = set(model["vocabulary"])
    features = tokenize(question)
    scores = {}
    for label in model["classes"]:
        probs = model["token_log_likelihood"][label]
        score = model["class_log_prior"][label]
        for feature in features:
            score += probs.get(feature, probs["<UNK>"] if feature not in vocab else 0.0)
        scores[label] = score
    peak = max(scores.values())
    denom = sum(math.exp(value - peak) for value in scores.values())
    normalized = {key: math.exp(value - peak) / denom for key, value in scores.items()}
    route = max(normalized, key=normalized.get)
    return {
        "route": route,
        "role_description": model["classes"][route],
        "relative_scores_not_calibrated_probabilities": normalized,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    print(json.dumps(predict(args.question, model), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
