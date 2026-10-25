from doc_to_atom.evaluation.metrics import (
    exact_match,
    first_line,
    is_refusal,
    refusal_f1,
    rouge_l,
    squad_normalize,
    word_f1,
)


def test_metrics():
    assert word_f1("Paris France", "Paris") > 0
    assert rouge_l("a b", "a b") == 1
    assert refusal_f1("unanswerable") == 1


def test_squad_normalization_removes_articles_ascii_punctuation_and_extra_space():
    assert squad_normalize("  The, PARIS!  ") == "paris"
    assert squad_normalize("A golden-anniversary") == "goldenanniversary"
    assert word_f1("The Paris", "Paris") == 1.0
    assert exact_match("An Eiffel Tower!", "Eiffel Tower") == 1.0


def test_first_line_and_refusal_use_first_nonempty_answer_line():
    assert first_line("\n  unanswerable.\nExplanation: absent") == "unanswerable."
    assert is_refusal("\n  unanswerable.\nExplanation: absent") is True
    assert is_refusal("This is not unanswerable") is False
