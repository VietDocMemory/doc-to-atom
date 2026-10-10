from __future__ import annotations

import re
import unicodedata
from collections import Counter

from rouge_score import rouge_scorer


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"\w+", text, flags=re.UNICODE)


def word_f1(prediction: str, gold: str) -> float:
    pred, target = _tokens(prediction), _tokens(gold)
    if not pred and not target:
        return 1.0
    overlap = sum((Counter(pred) & Counter(target)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(target)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, gold: str) -> float:
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False).score(gold, prediction)["rougeL"].fmeasure


def refusal_f1(prediction: str, target: str = "unanswerable") -> float:
    return word_f1(prediction, target)

