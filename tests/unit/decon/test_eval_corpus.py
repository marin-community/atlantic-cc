# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval-corpus rebuild: text policies (ported from Marin's test_prepare_eval_corpus.py), validation sets,
and the index/bloom builder with digests."""

import io
import json
import sys
import types
import zipfile
from dataclasses import replace
from importlib import resources

import dupekit_native
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atlantic_cc.decon import eval_corpus, val_sets
from atlantic_cc.decon.bloom import PyBloom
from atlantic_cc.decon.eval_corpus import (
    _PASSAGE_FIELDS,
    AA_EVALS,
    AAEvalConfig,
    AARecordType,
    _aa_manifest_entry,
    _aa_records,
    _iter_aa_rows,
    _lmh_doc_text,
)
from atlantic_cc.decon.eval_index import (
    build_eval_index,
    compare_file_digests,
    compare_record_digests,
    corpus_digests,
    eval_record_files,
    record_hashes,
)

_PASSAGE = "The rain had continued for a week and a flood created a big river by the farm."


def _prompt(doc: dict) -> str:
    body = " ".join(str(doc.get(k, "")) for k in ("article", "passage", "context", "premise", "story"))
    return f"Read: {body}\nQuestion: {doc.get('question', '')}"


def _target(doc: dict) -> str:
    return str(doc.get("answer", ""))


def test_lmh_passage_doc_drops_passage_keeps_qa():
    doc = {"article": _PASSAGE, "question": "What did Nancy do", "answer": "C", "options": ["ran", "gathered her cows"]}
    text = _lmh_doc_text(doc, _prompt, _target)
    assert _PASSAGE not in text
    assert text == "C\n\nC\n\nran\n\ngathered her cows\n\nWhat did Nancy do"  # target, then raw fields


def test_lmh_non_passage_doc_keeps_prompt_target_and_fields_in_sorted_key_order():
    doc = {"question": "What is two and two", "answer": "4", "id": 7}
    assert (
        _lmh_doc_text(doc, _prompt, _target)
        == "Read:     \nQuestion: What is two and two\n\n4\n\n4\n\nWhat is two and two"
    )


@pytest.mark.parametrize("field", sorted(_PASSAGE_FIELDS))
def test_every_passage_field_suppresses_the_passage(field):
    doc = {field.upper(): "UNIQUE_PASSAGE spanning words", "question": "Q_MARKER", "answer": "yes"}
    text = _lmh_doc_text(doc, lambda d: f"renders {d[field.upper()]}", _target)
    assert "UNIQUE_PASSAGE" not in text and "Q_MARKER" in text and "yes" in text


def test_failing_templates_and_falsy_targets_are_dropped():
    def boom(_doc):
        raise RuntimeError("no template")

    assert _lmh_doc_text({"question": "Q here", "answer": 0}, boom, lambda d: d["answer"]) == "Q here"


def _aa_config(**changes) -> AAEvalConfig:
    values = {
        "name": "Example Eval",
        "subdir": "example",
        "source_revision": "0123456789abcdef",
        "expected_records": 1,
        "official_records": 1,
        "hf_id": "owner/example",
        "subset": None,
        "split": "test",
        "text_fields": ("question", "answer"),
    }
    return AAEvalConfig(**(values | changes))


def test_aa_configs_equal_the_as_run_manifest():
    with resources.files("atlantic_cc.decon").joinpath("data/as_run/aa_manifest.json").open() as fh:
        manifest = json.load(fh)
    assert [_aa_manifest_entry(cfg) for cfg in AA_EVALS] == manifest["benchmarks"]
    assert sum(cfg.expected_records for cfg in AA_EVALS) == 4_165


def test_aa_hf_loader_uses_the_pinned_revision(monkeypatch):
    calls = []
    fake = types.ModuleType("datasets")
    fake.Image = type("Image", (), {})
    fake.load_dataset = lambda dataset_id, *, name, split, revision: (
        calls.append((dataset_id, name, split, revision)) or [{"question": "Question", "answer": "Answer"}]
    )
    monkeypatch.setitem(sys.modules, "datasets", fake)
    assert list(_iter_aa_rows(_aa_config())) == [{"question": "Question", "answer": "Answer"}]
    assert calls == [("owner/example", None, "test", "0123456789abcdef")]


def test_hle_indexes_question_and_answer_of_multimodal_rows():
    hle = replace(next(c for c in AA_EVALS if c.subdir == "hle"), expected_records=1)
    [record] = _aa_records([{"id": "img", "question": "Q with figure", "answer": "A", "image": {"bytes": b"x"}}], hle)
    assert record == {"id": "hle-img", "text": "Q with figure\n\nA"}


def test_scicode_expands_each_subproblem():
    cfg = _aa_config(subdir="scicode", expected_records=2, record_type=AARecordType.SCICODE_SUBPROBLEM, text_fields=())
    steps = [
        {"step_number": "7.1", "step_description_prompt": "First", "function_header": "def f():", "test_cases": ["t1"]},
        {"step_number": "7.2", "step_description_prompt": "Second", "return_line": "return 2"},
    ]
    records = _aa_records([{"problem_description_main": "Main", "sub_steps": steps}], cfg)
    assert records == [
        {"id": "scicode-7.1", "text": "Main\n\nFirst\n\ndef f():\n\nt1"},
        {"id": "scicode-7.2", "text": "Main\n\nSecond\n\nreturn 2"},
    ]


def test_tau3_indexes_instructions_and_sorted_criteria():
    cfg = _aa_config(subdir="tau3_banking", record_type=AARecordType.TAU3_TASK, text_fields=())
    rows = [
        {"id": "t1", "user_scenario": {"instructions": "Ask for a card."}, "evaluation_criteria": {"b": 1, "a": "é"}}
    ]
    assert _aa_records(rows, cfg) == [{"id": "tau3_banking-t1", "text": 'Ask for a card.\n\n{"a": "é", "b": 1}'}]


def test_terminal_bench_archive_pairs_instructions_with_solutions(monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("tb/tasks/task-b/instruction.md", "Instruction B")
        zf.writestr("tb/tasks/task-b/solution/solve.sh", "Solution B")
        zf.writestr("tb/tasks/task-a/instruction.md", "Instruction A")
        zf.writestr("tb/tasks/task-a/solution/solve.sh", "Solution A")
    archive.seek(0)
    monkeypatch.setattr(eval_corpus.urllib.request, "urlopen", lambda _url: archive)
    cfg = _aa_config(
        subdir="terminal_bench_2_1",
        source_url="https://example.test/archive.zip",
        expected_records=2,
        record_type=AARecordType.TERMINAL_BENCH_TASK,
        hf_id=None,
        text_fields=("instruction", "solution"),
    )
    assert _aa_records(list(_iter_aa_rows(cfg)), cfg) == [
        {"id": "terminal_bench_2_1-task-a", "text": "Instruction A\n\nSolution A"},
        {"id": "terminal_bench_2_1-task-b", "text": "Instruction B\n\nSolution B"},
    ]


def test_aa_record_count_mismatch_stops_preparation():
    with pytest.raises(ValueError, match="Example Eval: expected 2 records, extracted 1"):
        _aa_records([{"question": "Only question", "answer": "Only answer"}], _aa_config(expected_records=2))


def test_fill_special_cases():
    assert eval_corpus.lmh_load_name("include_base_44_arabic_few_shot_og") == "include_base_44_arabic"
    assert eval_corpus.mmmlu_config("mmmlu_pt_br") == "PT_BR" and eval_corpus.mmmlu_config("mmlu") is None
    row = {"Question": "Q?", "A": "a", "B": "b", "C": "c", "D": "d", "Answer": "C"}
    assert eval_corpus.mmmlu_text(row) == "Q?\nA. a\nB. b\nC. c\nD. d\nAnswer: c"
    squad = {
        "id": "s1",
        "title": "T",
        "context": "PASSAGE",
        "question": "Q",
        "answers": {"text": [], "answer_start": []},
    }
    assert eval_corpus.squad_v2_text(squad) == "unanswerable\n\ns1\n\nQ\n\nT"
    assert len(eval_corpus.configured_lmh_tasks()) == 339


def test_validation_set_text_extraction():
    assert len(val_sets.val_set_names()) == 24
    assert val_sets.uncheatable_text({"title": " T ", "body": " B "}) == " B "
    assert val_sets.uncheatable_text({"title": " T ", "body": "  "}) == "T\n\n"
    assert val_sets.uncheatable_text({"paragraphs": ["a", None, "b"]}) == "a\nb"
    assert val_sets.uncheatable_text({"title": "T", "x": 1}) == "T"
    assert val_sets.uncheatable_id({"metadata": {"uuid": "u"}}, "bbc_news", 3) == "u"
    assert val_sets.uncheatable_id({}, "bbc_news", 3) == "bbc_news_20250901to20250914_000003"
    assert val_sets.lima_text({"conversations": ["hi", "hello"]}) == "User: hi\n\nAssistant: hello"
    assert val_sets.lima_text({"conversations": []}) is None


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema([("id", pa.string()), ("text", pa.string())])), path)


def test_index_bloom_and_digests_from_a_staged_corpus(tmp_path):
    long = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    _write(tmp_path / "full/aa/hle/test.parquet", [{"id": "hle-1", "text": long}, {"id": "hle-2", "text": ""}])
    _write(tmp_path / "full/lmh/gsm8k/eval.parquet", [{"id": "gsm8k-test-0", "text": "Lana Del Rey sings"}])
    _write(tmp_path / "full/lmh/wikitext/eval.parquet", [{"id": "wikitext-test-0", "text": "excluded task text here"}])
    _write(tmp_path / "val/lima/eval.parquet", [{"id": "lima-0", "text": "User: hi there friend\n\nAssistant: hello"}])
    assert [p.split("full/")[1] for p in eval_record_files(str(tmp_path / "full"))] == [
        "aa/hle/test.parquet",
        "lmh/gsm8k/eval.parquet",
    ]

    report = build_eval_index(str(tmp_path / "full"), str(tmp_path / "val"), str(tmp_path / "index"))
    assert (report["full_index_rows"], report["val_index_rows"]) == (3, 1)
    assert report["capacity"] == 5  # int((3 + 1) * 1.25)
    index = pq.read_table(tmp_path / "index" / "eval_hash_index.parquet").to_pylist()
    assert {(r["hash"], r["eval_id"]) for r in index} == {(h, "hle-1") for h in record_hashes(long)} | {
        (h, "gsm8k-test-0") for h in record_hashes("Lana Del Rey sings")
    }
    bloom = PyBloom.open(str(tmp_path / "index" / "filter.bin"))
    native = dupekit_native.Bloom.load(str(tmp_path / "index" / "filter.bin"))
    for row in index:
        assert row["hash"] in bloom and row["hash"] in native

    summaries, table = corpus_digests(str(tmp_path / "full"), str(tmp_path / "val"))
    assert [s["file"] for s in summaries] == [
        "full/aa/hle/test.parquet",
        "full/lmh/gsm8k/eval.parquet",
        "val/lima/eval.parquet",
    ]
    assert table.num_rows == 3
    _write(tmp_path / "full/lmh/gsm8k/eval.parquet", [{"id": "gsm8k-test-0", "text": "Lana Del Rey dances"}])
    rebuilt, _ = corpus_digests(str(tmp_path / "full"), str(tmp_path / "val"))
    diff = compare_file_digests(rebuilt, summaries)
    assert diff["records_differ"] == diff["features_differ"] == ["full/lmh/gsm8k/eval.parquet"]
    assert diff["missing"] == diff["extra"] == []
    _, rebuilt_table = corpus_digests(str(tmp_path / "full"), str(tmp_path / "val"))
    records = compare_record_digests(rebuilt_table, table)
    assert (records["changed"], records["missing"], records["extra"]) == (1, 0, 0)
    assert records["examples_changed"] == [["full/lmh/gsm8k/eval.parquet", "gsm8k-test-0"]]


def test_shipped_file_digests_cover_the_as_run_corpus():
    with resources.files("atlantic_cc.decon").joinpath("data/as_run/eval_file_digests.json").open() as fh:
        digests = json.load(fh)
    assert len(digests) == 1_147
    assert sum(d["records"] for d in digests) == 2_106_579
    assert sum(d["index_rows"] for d in digests) == 133_669_639
