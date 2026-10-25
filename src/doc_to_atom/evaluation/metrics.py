from __future__ import annotations

import re
import string
from collections import Counter

from rouge_score import rouge_scorer


def squad_normalize(text: str) -> str:
    """Match the official English SQuAD lower/punctuation/article/space normalization."""
    lowered = text.lower()
    without_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _tokens(text: str) -> list[str]:
    return squad_normalize(text).split()


def exact_match(prediction: str, gold: str) -> float:
    return float(squad_normalize(prediction) == squad_normalize(gold))


def first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def is_refusal(prediction: str, literal: str = "unanswerable") -> bool:
    target = squad_normalize(literal)
    return bool(target) and squad_normalize(first_line(prediction)) == target


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
    return (
        rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        .score(gold, prediction)["rougeL"]
        .fmeasure
    )


def refusal_f1(prediction: str, target: str = "unanswerable") -> float:
    return float(is_refusal(prediction, target))
