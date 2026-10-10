from doc_to_atom.evaluation.metrics import refusal_f1, rouge_l, word_f1


def test_metrics():
    assert word_f1("Paris France", "Paris") > 0
    assert rouge_l("a b", "a b") == 1
    assert refusal_f1("unanswerable") == 1
