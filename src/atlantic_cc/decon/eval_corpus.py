# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the ``full`` eval corpus: AA Intelligence Index v4.1.1 benchmarks and lm-eval-harness tasks.

Best effort. Ports Marin ``experiments/datakit/decontam/prepare_eval_corpus.py`` and
``experiments/fast_curation/decon_eval_corpus_fill.py`` (tag ``lpv11-8m-as-run``). The as-run corpus
``aa-index-v4.1.1-v3-fill`` was staged in two environments (809 lm-eval leaves under ``datasets`` 4.8.5,
then 305 restaged leaves under the lm-eval fork's ``datasets`` 3.6); this rebuild stages every configured
task in one environment with the fill's fixes (legacy dataset ids, ``trust_remote_code``, MMMLU from
``openai/MMMLU``, squad_v2 from parquet, ``include_base_44_<lang>`` for the unregistered ``_few_shot_og``
names). lm-eval datasets were not revision-pinned (Hub ``main`` on 2026-10-19/20), several AA sources are
gated (GPQA, HLE) and three AA pins have moved upstream, so compare a rebuild with the shipped digests
(``atlantic-cc decon eval-digests``) rather than assuming equality.

Layout written under ``root``: ``aa/<subdir>/<split>.parquet`` and ``lmh/<task>/eval.parquet``, each with
``id, text`` records, plus ``aa/_manifest.json`` and ``lmh/_manifest.json``.
"""

import dataclasses
import io as io_mod
import json
import logging
import re
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Iterator
from enum import StrEnum
from importlib import resources
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.decon.lmh_loader import flatten_task_dict, materialize_first_nonempty_split, trust_remote_code_for_hf

logger = logging.getLogger(__name__)

AA_INDEX_VERSION = "4.1.1"
EVAL_CORPUS_VERSION = "aa-index-v4.1.1-v3"
LMH_EXTRACTION_VERSION = "2-passage-drop"
AS_RUN_LMH_MANIFEST = "data/as_run/lmh_manifest.json"
AS_RUN_AA_MANIFEST = "data/as_run/aa_manifest.json"
MMMLU_REPO = "openai/MMMLU"
SQUAD_V2_PARQUET = "hf://datasets/rajpurkar/squad_v2/squad_v2/validation-*.parquet"
_INCLUDE_RE = re.compile(r"^(include_base_44_.+)_few_shot_og$")
_MMMLU_RE = re.compile(r"^mmmlu_([a-z]{2})_([a-z]{2})$")


class AARecordType(StrEnum):
    """AA source-row conversion policy."""

    ROW = "row"
    SCICODE_SUBPROBLEM = "scicode_subproblem"
    TAU3_TASK = "tau3_task"
    TERMINAL_BENCH_TASK = "terminal_bench_task"


@dataclasses.dataclass(frozen=True)
class AAEvalConfig:
    name: str
    subdir: str
    source_revision: str
    expected_records: int
    official_records: int
    hf_id: str | None
    subset: str | None
    split: str
    source_url: str | None = None
    record_type: AARecordType = AARecordType.ROW
    text_fields: tuple[str, ...] = ()
    list_fields: tuple[str, ...] = ()
    skip_if: Callable[[dict], bool] | None = None
    coverage_note: str = "Full official task set."
    filter_note: str | None = None


AA_EVALS: tuple[AAEvalConfig, ...] = (
    AAEvalConfig(
        name="GDPval-AA v2",
        subdir="gdpval_aa_v2",
        source_revision="11e7900cdcac61bc4daf59e65feb238acda98fbf",
        expected_records=220,
        official_records=220,
        hf_id="openai/gdpval",
        subset=None,
        split="train",
        text_fields=("prompt", "rubric_pretty"),
        coverage_note="All public GDPval tasks; prompt and hidden grading rubric are indexed.",
    ),
    AAEvalConfig(
        name="tau3-Banking",
        subdir="tau3_banking",
        source_revision="fc0055dc4e0a316c3f83133267fbd6faaa770992",
        expected_records=97,
        official_records=97,
        hf_id=None,
        subset=None,
        split="test",
        source_url=(
            "https://raw.githubusercontent.com/sierra-research/tau2-bench/"
            "fc0055dc4e0a316c3f83133267fbd6faaa770992/data/tau2/domains/banking_knowledge/tasks.json"
        ),
        record_type=AARecordType.TAU3_TASK,
        coverage_note="Full upstream v1.0.1 task set; user scenarios and hidden evaluation criteria are indexed.",
    ),
    AAEvalConfig(
        name="Terminal-Bench v2.1",
        subdir="terminal_bench_2_1",
        source_revision="c5ee500c185224c97cd6caff7866a990a0057f41",
        expected_records=89,
        official_records=89,
        hf_id=None,
        subset=None,
        split="test",
        source_url=(
            "https://github.com/harbor-framework/terminal-bench-2-1/archive/c5ee500c185224c97cd6caff7866a990a0057f41.zip"
        ),
        record_type=AARecordType.TERMINAL_BENCH_TASK,
        text_fields=("instruction", "solution"),
        coverage_note="All 89 task instructions and oracle solutions from the pinned v2.1 task tree.",
    ),
    AAEvalConfig(
        name="SciCode",
        subdir="scicode",
        source_revision="4510f6a6aa27c43fad7b43da2c59602a86e88480",
        expected_records=291,
        official_records=288,
        hf_id="SciCode1/SciCode",
        subset=None,
        split="test",
        record_type=AARecordType.SCICODE_SUBPROBLEM,
        coverage_note=(
            "Pinned public test split. It contains 291 subproblems, three more than the 288 reported by AA; "
            "the conservative superset is indexed."
        ),
    ),
    AAEvalConfig(
        name="AA-LCR",
        subdir="aa_lcr",
        source_revision="bdae010bbce259820c0e34c1d7cce210d966fb75",
        expected_records=100,
        official_records=100,
        hf_id="ArtificialAnalysis/AA-LCR",
        subset=None,
        split="test",
        text_fields=("question", "answer"),
        coverage_note=(
            "All public questions and answers. Public source documents are prompt context and are not indexed."
        ),
    ),
    AAEvalConfig(
        name="AA-Omniscience",
        subdir="aa_omniscience",
        source_revision="4a8ffc87c4650054825fb767fe0da4a4fc97ff32",
        expected_records=600,
        official_records=6000,
        hf_id="ArtificialAnalysis/AA-Omniscience-Public",
        subset=None,
        split="train",
        text_fields=("question", "answer"),
        coverage_note="The mandatory public release contains 600 of the 6,000 private evaluation questions.",
    ),
    AAEvalConfig(
        name="Humanity's Last Exam",
        subdir="hle",
        source_revision="5a81a4c7271a2a2a312b9a690f0c2fde837e4c29",
        expected_records=2500,
        official_records=2500,
        hf_id="cais/hle",
        subset=None,
        split="test",
        text_fields=("question", "answer"),
        coverage_note=(
            "All 2,500 questions from the May 2025 revision. For multimodal rows, the question and answer text "
            "are indexed; image bytes are not."
        ),
    ),
    AAEvalConfig(
        name="GPQA Diamond",
        subdir="gpqa_diamond",
        source_revision="633f5ee89ab8ad4522a9f850766b73f62147ffdd",
        expected_records=198,
        official_records=198,
        hf_id="Idavidrein/gpqa",
        subset="gpqa_diamond",
        split="train",
        text_fields=("Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"),
    ),
    AAEvalConfig(
        name="CritPt",
        subdir="critpt",
        source_revision="9b9fc8498596ec08ab5437a72f4aa18beef2b876",
        expected_records=70,
        official_records=70,
        hf_id="CritPt-Benchmark/CritPt",
        subset=None,
        split="train",
        text_fields=("problem_description", "code_template", "answer_code", "answer_only_code"),
        coverage_note=(
            "All 70 public composite challenges used by AA, with response templates and published answer code. "
            "CritPt reports 189 modular checkpoints for the 70 test challenges, but those checkpoint prompts are "
            "not part of the public dataset."
        ),
    ),
)


def _extract_aa_text(row: dict[str, Any], cfg: AAEvalConfig) -> str:
    """Pin-named extraction first; fall back to generic concat when nothing matches."""
    parts: list[str] = []
    for field in cfg.text_fields:
        v = row.get(field)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    for field in cfg.list_fields:
        v = row.get(field)
        if isinstance(v, list):
            parts.extend(s for s in v if isinstance(s, str) and s.strip())
    if parts:
        return "\n\n".join(parts)
    return _concat_strings(row)


def _concat_strings(record: dict[str, Any], exclude: frozenset[str] = frozenset()) -> str:
    """Concat all string-typed fields in sorted key order; flatten list[str]. Lowercase keys in *exclude* skip."""
    parts: list[str] = []
    for k in sorted(record.keys()):
        if k.lower() in exclude:
            continue
        v = record[k]
        if isinstance(v, str) and v.strip():
            parts.append(v)
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            parts.extend(s for s in v if s.strip())
    return "\n\n".join(parts)


# Passage-bearing reading-comprehension docs index only question + answer: the public passage would flag
# every web page quoting it.
_PASSAGE_FIELDS: frozenset[str] = frozenset(
    {"passage", "context", "ctx", "article", "story", "premise", "background", "document", "paragraph", "support"}
)


def _lmh_doc_text(doc: Any, prompt_fn: Callable, target_fn: Callable) -> str:
    """Indexed eval text for one lm-eval-harness doc (extraction version ``2-passage-drop``)."""
    has_passage = isinstance(doc, dict) and any(k.lower() in _PASSAGE_FIELDS for k in doc)
    parts: list[str] = []
    if not has_passage:
        try:
            prompt = prompt_fn(doc) or ""
        except Exception:
            prompt = ""
        if prompt:
            parts.append(str(prompt))
    try:
        target = target_fn(doc) or ""
    except Exception:
        target = ""
    if target:
        parts.append(str(target))
    if isinstance(doc, dict):
        parts.append(_concat_strings(doc, exclude=_PASSAGE_FIELDS if has_passage else frozenset()))
    return "\n\n".join(p for p in parts if p.strip())


_PARQUET_SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string())])
_PARQUET_BATCH = 1000


def write_records(path: str, records: Iterable[dict]) -> int:
    """Write ``{id, text}`` records to one zstd parquet file in 1,000-row batches; returns the row count."""
    n = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []
    with fsspec.open(path, "wb") as raw:
        writer = pq.ParquetWriter(raw, _PARQUET_SCHEMA, compression="zstd")
        try:
            for rec in records:
                batch_ids.append(rec["id"])
                batch_texts.append(rec["text"])
                if len(batch_ids) >= _PARQUET_BATCH:
                    writer.write_table(pa.table({"id": batch_ids, "text": batch_texts}, schema=_PARQUET_SCHEMA))
                    n += len(batch_ids)
                    batch_ids, batch_texts = [], []
            if batch_ids:
                writer.write_table(pa.table({"id": batch_ids, "text": batch_texts}, schema=_PARQUET_SCHEMA))
                n += len(batch_ids)
        finally:
            writer.close()
    return n


def _iter_aa_rows(cfg: AAEvalConfig) -> Iterator[dict[str, Any]]:
    """Stream raw rows for one pinned AA source."""
    if cfg.record_type == AARecordType.TERMINAL_BENCH_TASK:
        if cfg.source_url is None:
            raise ValueError(f"{cfg.name}: source_url is required")
        with urllib.request.urlopen(cfg.source_url) as response:
            archive = response.read()
        with zipfile.ZipFile(io_mod.BytesIO(archive)) as source_zip:
            instructions: dict[str, str] = {}
            solutions: dict[str, str] = {}
            for path in source_zip.namelist():
                if "/tasks/" not in path:
                    continue
                task_path = path.split("/tasks/", 1)[1]
                task_name, separator, relative_path = task_path.partition("/")
                if not separator:
                    continue
                if relative_path == "instruction.md":
                    instructions[task_name] = source_zip.read(path).decode("utf-8")
                elif relative_path == "solution/solve.sh":
                    solutions[task_name] = source_zip.read(path).decode("utf-8")
        if instructions.keys() != solutions.keys():
            raise ValueError(
                f"{cfg.name}: task archive is incomplete; missing instructions="
                f"{sorted(solutions.keys() - instructions.keys())}, missing solutions="
                f"{sorted(instructions.keys() - solutions.keys())}"
            )
        for task_name in sorted(instructions):
            yield {"task_name": task_name, "instruction": instructions[task_name], "solution": solutions[task_name]}
        return

    if cfg.source_url is not None:
        with urllib.request.urlopen(cfg.source_url) as response:
            rows = json.load(response)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"{cfg.name}: expected a JSON list of objects")
        yield from rows
        return

    if cfg.hf_id is None:
        raise ValueError(f"{cfg.name}: hf_id or source_url is required")
    from datasets import Image as DatasetsImage  # noqa: PLC0415  (optional rebuild dependency)
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(cfg.hf_id, name=cfg.subset, split=cfg.split, revision=cfg.source_revision)
    # Do not decode HLE image bytes; question and answer text are what gets indexed.
    features = getattr(ds, "features", None) or {}
    for col, ftype in features.items():
        if isinstance(ftype, DatasetsImage):
            ds = ds.cast_column(col, DatasetsImage(decode=False))
    for row in ds:
        yield dict(row)


def _row_identifier(row: dict[str, Any], index: int) -> str:
    for field in ("id", "question_id", "problem_id"):
        value = row.get(field)
        if isinstance(value, str | int):
            return str(value)
    return str(index)


def _scicode_text(row: dict[str, Any], subproblem: dict[str, Any]) -> str:
    parts = [
        row.get("problem_description_main"),
        row.get("problem_background_main"),
        row.get("required_dependencies"),
        subproblem.get("step_description_prompt"),
        subproblem.get("step_background"),
        subproblem.get("function_header"),
        *(subproblem.get("test_cases") or []),
        subproblem.get("return_line"),
        subproblem.get("ground_truth_code"),
    ]
    return "\n\n".join(part for part in parts if isinstance(part, str) and part.strip())


def _aa_records(raw_rows: Iterable[dict[str, Any]], cfg: AAEvalConfig) -> list[dict[str, str]]:
    """Convert one AA source to validated decontamination records."""
    records: list[dict[str, str]] = []
    for index, row in enumerate(raw_rows):
        if cfg.skip_if is not None and cfg.skip_if(row):
            continue

        if cfg.record_type == AARecordType.SCICODE_SUBPROBLEM:
            subproblems = row.get("sub_steps")
            if not isinstance(subproblems, list):
                raise ValueError(f"{cfg.name}: row {_row_identifier(row, index)} has no sub_steps list")
            for subproblem in subproblems:
                if not isinstance(subproblem, dict):
                    raise ValueError(f"{cfg.name}: row {_row_identifier(row, index)} has an invalid subproblem")
                subproblem_id = subproblem.get("step_number")
                if not isinstance(subproblem_id, str) or not subproblem_id:
                    raise ValueError(f"{cfg.name}: row {_row_identifier(row, index)} has a subproblem without an ID")
                text = _scicode_text(row, subproblem)
                if not text:
                    raise ValueError(f"{cfg.name}: subproblem {subproblem_id} has no indexed text")
                records.append({"id": f"{cfg.subdir}-{subproblem_id}", "text": text})
            continue

        if cfg.record_type == AARecordType.TAU3_TASK:
            task_id = row.get("id")
            scenario = row.get("user_scenario")
            criteria = row.get("evaluation_criteria")
            if not isinstance(task_id, str) or not isinstance(scenario, dict) or not isinstance(criteria, dict):
                raise ValueError(f"{cfg.name}: task {index} has an invalid schema")
            instructions = scenario.get("instructions")
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError(f"{cfg.name}: task {task_id} has no user instructions")
            text = f"{instructions}\n\n{json.dumps(criteria, sort_keys=True, ensure_ascii=False)}"
            records.append({"id": f"{cfg.subdir}-{task_id}", "text": text})
            continue

        text = _extract_aa_text(row, cfg)
        if not text:
            raise ValueError(f"{cfg.name}: row {_row_identifier(row, index)} has no indexed text")
        if cfg.record_type == AARecordType.TERMINAL_BENCH_TASK:
            identifier = row.get("task_name")
            if not isinstance(identifier, str):
                raise ValueError(f"{cfg.name}: task {index} has no task_name")
        else:
            identifier = _row_identifier(row, index)
        records.append({"id": f"{cfg.subdir}-{identifier}", "text": text})

    if len(records) != cfg.expected_records:
        raise ValueError(f"{cfg.name}: expected {cfg.expected_records} records, extracted {len(records)}")
    return records


def _aa_manifest_entry(cfg: AAEvalConfig) -> dict[str, Any]:
    source = cfg.hf_id if cfg.hf_id is not None else cfg.source_url
    assert source is not None
    return {
        "name": cfg.name,
        "artifact": f"{cfg.subdir}/{cfg.split}.parquet",
        "source": source,
        "source_revision": cfg.source_revision,
        "subset": cfg.subset,
        "split": cfg.split,
        "record_type": cfg.record_type.value,
        "text_fields": list(cfg.text_fields),
        "list_fields": list(cfg.list_fields),
        "expected_records": cfg.expected_records,
        "official_records": cfg.official_records,
        "coverage_note": cfg.coverage_note,
        "filter_note": cfg.filter_note,
    }


def _write_json(path: str, value: dict[str, Any]) -> None:
    with fsspec.open(path, "w") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def prepare_aa(root: str) -> dict[str, Any]:
    """Stage the nine AA benchmarks at their pinned revisions (fails if any count differs)."""
    for cfg in AA_EVALS:
        out_path = f"{root}/aa/{cfg.subdir}/{cfg.split}.parquet"
        n = write_records(out_path, _aa_records(_iter_aa_rows(cfg), cfg))
        logger.info("aa/%s: %d records -> %s", cfg.subdir, n, out_path)
    manifest = {
        "schema_version": 1,
        "corpus_version": f"{EVAL_CORPUS_VERSION}-fill",
        "suite": f"Artificial Analysis Intelligence Index v{AA_INDEX_VERSION}",
        "required": True,
        "status": "complete",
        "benchmarks": [_aa_manifest_entry(cfg) for cfg in AA_EVALS],
    }
    _write_json(f"{root}/aa/_manifest.json", manifest)
    return manifest


def as_run_lmh_manifest() -> dict[str, Any]:
    """The as-run ``lmh/_manifest.json`` of ``aa-index-v4.1.1-v3-fill`` (task names and record counts only)."""
    with resources.files("atlantic_cc.decon").joinpath(AS_RUN_LMH_MANIFEST).open("r") as fh:
        return json.load(fh)


def configured_lmh_tasks() -> list[str]:
    """The 339 configured lm-eval task names (union of Marin's eval bundles minus the excluded tasks)."""
    return list(as_run_lmh_manifest()["configured_tasks"])


def lmh_load_name(task: str) -> str:
    """lm-eval task to load for a configured task the pinned lm-eval does not register."""
    m = _INCLUDE_RE.match(task)
    return m.group(1) if m else task


def mmmlu_config(task: str) -> str | None:
    """``openai/MMMLU`` config for an ``mmmlu_<lang>_<region>`` task (``mmmlu_ar_xy`` -> ``AR_XY``)."""
    m = _MMMLU_RE.match(task)
    return f"{m.group(1)}_{m.group(2)}".upper() if m else None


def mmmlu_text(row: dict) -> str:
    choices = "\n".join(f"{k}. {row[k]}" for k in ("A", "B", "C", "D"))
    return f"{row['Question']}\n{choices}\nAnswer: {row[row['Answer']]}"


def squad_v2_text(row: dict) -> str:
    """Indexed text for a squad_v2 row: the passage is dropped, as for every passage-bearing task."""
    answers = row["answers"]["text"]
    return _lmh_doc_text(row, lambda doc: "", lambda doc: answers[0] if answers else "unanswerable")


def _stage_mmmlu(task: str, config: str, root: str) -> list[dict]:
    from datasets import load_dataset  # noqa: PLC0415  (optional rebuild dependency)

    docs = load_dataset(MMMLU_REPO, config, split="test")
    rows = ({"id": f"{task}-test-{i}", "text": mmmlu_text(row)} for i, row in enumerate(docs))
    n = write_records(f"{root}/{task}/eval.parquet", rows)
    return [{"task": task, "artifact": f"{task}/eval.parquet", "records": n}]


def _stage_squad_v2(task: str, root: str) -> list[dict]:
    from datasets import load_dataset  # noqa: PLC0415  (optional rebuild dependency)

    docs = load_dataset("parquet", data_files=SQUAD_V2_PARQUET, split="train")
    rows = ({"id": f"{task}-validation-{i}", "text": squad_v2_text(row)} for i, row in enumerate(docs))
    n = write_records(f"{root}/{task}/eval.parquet", rows)
    return [{"task": task, "artifact": f"{task}/eval.parquet", "records": n}]


def _stage_lmh(task: str, root: str) -> list[dict]:
    from lm_eval.tasks import get_task_dict  # noqa: PLC0415  (optional rebuild dependency)

    entries = []
    for child, lm_task in flatten_task_dict(get_task_dict([lmh_load_name(task)])):
        chosen = materialize_first_nonempty_split(lm_task)
        if chosen is None:
            raise ValueError(f"{child}: no docs in any split")
        split, docs = chosen

        def rows(lm_task=lm_task, docs=docs, split=split, child=child) -> Iterator[dict]:
            for i, doc in enumerate(docs):
                text = _lmh_doc_text(doc, lm_task.doc_to_text, lm_task.doc_to_target)
                if text:
                    yield {"id": f"{child}-{split}-{i}", "text": text}

        n = write_records(f"{root}/{child}/eval.parquet", rows())
        entries.append({"task": child, "artifact": f"{child}/eval.parquet", "records": n})
    return entries


def prepare_lmh(root: str, tasks: list[str] | None = None) -> dict[str, Any]:
    """Stage every configured lm-eval task (or *tasks*) as leaf files under ``root/lmh``."""
    lmh_root = f"{root}/lmh"
    names = tasks or configured_lmh_tasks()
    trust_remote_code_for_hf()
    artifacts: list[dict] = []
    failed: list[dict] = []
    for task in names:
        try:
            config = mmmlu_config(task)
            if config:
                entries = _stage_mmmlu(task, config, lmh_root)
            elif task == "squadv2":
                entries = _stage_squad_v2(task, lmh_root)
            else:
                entries = _stage_lmh(task, lmh_root)
        except Exception as exc:
            logger.warning("lmh/%s: failed: %s", task, exc)
            failed.append({"task": task, "reason": str(exc)})
            continue
        logger.info("lmh/%s: %s", task, [(e["task"], e["records"]) for e in entries])
        artifacts.extend(entries)
    manifest = {
        "schema_version": 1,
        "corpus_version": f"{EVAL_CORPUS_VERSION}-fill",
        "required": False,
        "status": "complete_with_failures" if failed else "complete",
        "extraction_version": LMH_EXTRACTION_VERSION,
        "configured_tasks": names,
        "included_leaf_tasks": sorted({a["task"] for a in artifacts}),
        "artifacts": sorted(artifacts, key=lambda a: a["task"]),
        "failed": failed,
    }
    _write_json(f"{lmh_root}/_manifest.json", manifest)
    return manifest
