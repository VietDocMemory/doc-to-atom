import pytest

from doc_to_atom.data.benchmark_catalog import BENCHMARK_CATALOG, get_benchmark_dataset


def test_catalog_covers_all_longbench_rows_in_paper_table():
    expected = {
        "longbench_2wikimqa", "longbench_hotpotqa", "longbench_multifieldqa_en",
        "longbench_multifieldqa_zh", "longbench_musique", "longbench_narrativeqa",
        "longbench_qasper", "longbench_triviaqa",
    }
    assert expected <= set(BENCHMARK_CATALOG)
    assert all(BENCHMARK_CATALOG[name].hf_dataset == "THUDM/LongBench" for name in expected)


def test_unknown_catalog_alias_requires_pinned_custom_export():
    with pytest.raises(ValueError, match="source-jsonl"):
        get_benchmark_dataset("not-a-benchmark")
