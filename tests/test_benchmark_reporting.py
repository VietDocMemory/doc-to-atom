from doc_to_atom.evaluation.benchmark import render_markdown_table, score_rows, summarize_rows


def test_scores_multiple_references_and_renders_paper_columns():
    rows = score_rows(
        [
            {
                "model": "gemma",
                "method": "doc2atom",
                "dataset": "squad",
                "split": "test",
                "prediction": "Paris",
                "gold": ["Paris", "City of Paris"],
            }
        ]
    )
    assert rows[0]["f1"] == 1
    table = render_markdown_table(summarize_rows(rows), "f1")
    assert "Gemma" not in table  # Model labels stay caller-controlled.
    assert "gemma Doc2Atom" in table
    assert "100.00" in table
