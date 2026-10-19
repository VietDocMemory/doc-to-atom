from doc_to_atom.data import prepare


def test_squad_preparation_preserves_exact_answer_lists():
    rows = [
        {
            "id": "q1",
            "context": "Context.",
            "question": "Question?",
            "answers": {"text": ["Gold", "Gold", " gold! "]},
        }
    ]

    [record] = prepare.group_squad_rows(rows, "train")

    question = record["questions"][0]
    assert question["answers"] == ["Gold", "Gold", " gold! "]
    assert "answer" not in question


def test_prepare_allows_zero_rows_for_a_split(tmp_path, monkeypatch):
    rows = [
        {
            "id": "q1",
            "context": "Context.",
            "question": "Question?",
            "answers": {"text": ["Gold"]},
        }
    ]
    calls = []

    def fake_load_dataset(name, split):
        calls.append(split)
        return rows

    monkeypatch.setattr(prepare, "load_dataset", fake_load_dataset)

    report = prepare.prepare_squad_subset(tmp_path / "source.jsonl", 1, 0)

    assert calls == ["train[:1]"]
    assert report["records"] == 1
