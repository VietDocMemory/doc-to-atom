from doc_to_atom.training.tracking import flatten_scalars


def test_flatten_scalars_keeps_metrics_and_excludes_examples():
    flattened = flatten_scalars(
        {
            "step": 3,
            "train": {"overall": {"losses": {"lm": 1.25}, "examples": [{"x": 1}]}},
            "validation": None,
        }
    )
    assert flattened == {"step": 3, "train/overall/losses/lm": 1.25}
