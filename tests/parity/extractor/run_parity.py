# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Byte-for-byte parity of resiliparse-rs builds on every HTML page of Common Crawl fixture WARCs.

Each variant is an interpreter that imports some ``resiliparse._extract_rs`` (a wheel installed in a venv,
or production's unpacked artifact via ``--package-dir``). Every page goes through production's call in a
supervised worker process (``worker.py``): a page that kills the process is recorded as ``crash``, one
that stalls as ``timeout``, and the worker restarts after it. A variant passes when its status and text
sha256 equal the reference variant's on every page.

    python -m tests.parity.extractor.run_parity --work-dir /root/parity --fixture F1 --fixture F3 \\
        --variant asrun=/root/venvs/asrun/bin/python --variant source=/root/venvs/source/bin/python \\
        --reference asrun --jobs 8

``--reference-results DIR`` compares local variants against a reference run made elsewhere (its
``index_<F>.json.gz`` and ``results_<F>_<reference>.jsonl.gz``), after checking both runs decoded the same
HTML. ``--write-golden PATH`` writes the reference's per-page hashes for one fixture as a compact TSV.
"""

import argparse
import gzip
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from tests.parity.extractor.pages import FIXTURES, download, read_index, write_pages

logger = logging.getLogger(__name__)

WORKER = Path(__file__).with_name("worker.py")
POLL_SECONDS = 1.0
DEFAULT_PAGE_TIMEOUT = 600.0
MAX_EXAMPLES = 20
GOLDEN_HASH_CHARS = 16


@dataclass(frozen=True)
class Variant:
    name: str
    python: str
    package_dir: str | None = None
    preload: tuple[str, ...] = ()


@dataclass
class RangeResult:
    records: dict[int, dict] = field(default_factory=dict)
    headers: list[dict] = field(default_factory=list)


def worker_command(variant: Variant, pages: Path, index: Path, start: int, end: int, out: Path) -> list[str]:
    command = [variant.python, "-I", str(WORKER), str(pages), str(index), str(start), str(end), str(out)]
    if variant.package_dir:
        command += ["--package-dir", variant.package_dir]
    for module in variant.preload:
        command += ["--preload", module]
    return command


def read_lines_from(path: Path, offset: int) -> list[dict]:
    """JSON lines written after byte ``offset``; a line cut short by a crash is ignored."""
    with open(path, "rb") as f:
        f.seek(offset)
        return [json.loads(line) for line in f.read().decode("utf-8").splitlines() if line.endswith("}")]


def run_range(variant: Variant, pages: Path, index: Path, start: int, end: int, out: Path, page_timeout: float):
    """Supervise workers over pages ``[start, end)`` until each page has a record; see the module docstring."""
    result = RangeResult()
    out.write_text("")
    log = out.with_suffix(".stderr")
    page = start
    while page < end:
        offset = out.stat().st_size
        with open(log, "ab") as stderr:
            process = subprocess.Popen(worker_command(variant, pages, index, page, end, out), stderr=stderr)
            last_size, last_change = offset, time.monotonic()
            while process.poll() is None:
                time.sleep(POLL_SECONDS)
                size = out.stat().st_size
                if size != last_size:
                    last_size, last_change = size, time.monotonic()
                elif time.monotonic() - last_change > page_timeout:
                    process.kill()
                    process.wait()
                    break
        lines = read_lines_from(out, offset)
        if not lines or not lines[0].get("header"):
            raise RuntimeError(f"{variant.name}: worker failed before extracting anything; see {log}")
        result.headers.append(lines[0])
        for record in lines[1:]:
            result.records[record["i"]] = record
        page += len(lines) - 1
        if page >= end:
            if process.returncode != 0:
                raise RuntimeError(f"{variant.name}: worker exited {process.returncode} after its last page")
            break
        status = "timeout" if process.returncode == -9 and time.monotonic() - last_change > page_timeout else "crash"
        result.records[page] = {"i": page, "status": status, "returncode": process.returncode}
        logger.warning("%s: page %d -> %s (exit %s)", variant.name, page, status, process.returncode)
        page += 1
    return result


def shard_bounds(n: int, shards: int) -> list[tuple[int, int]]:
    step = -(-n // shards)
    return [(lo, min(n, lo + step)) for lo in range(0, n, step)]


def run_variants(
    variants: list[Variant], fixture: str, pages: Path, index: Path, n_pages: int, work: Path, jobs: int, timeout: float
) -> dict[str, RangeResult]:
    shards = shard_bounds(n_pages, max(1, jobs))
    tasks = [(v, lo, hi) for v in variants for lo, hi in shards]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            (v.name, lo): pool.submit(
                run_range, v, pages, index, lo, hi, work / f"out_{fixture}_{v.name}_{lo}.jsonl", timeout
            )
            for v, lo, hi in tasks
        }
    merged: dict[str, RangeResult] = {v.name: RangeResult() for v in variants}
    for (name, _), future in sorted(futures.items()):
        part = future.result()
        merged[name].records.update(part.records)
        merged[name].headers.extend(part.headers)
    return merged


def page_key(record: dict) -> tuple:
    return (record["status"], record.get("sha256"), record.get("error"))


def summarize(records: dict[int, dict], n_pages: int) -> dict:
    counts: dict[str, int] = {}
    for record in records.values():
        label = record["status"] if record["status"] != "exception" else f"exception:{record['error']}"
        counts[label] = counts.get(label, 0) + 1
    empty = sum(1 for r in records.values() if r["status"] == "ok" and r["chars"] == 0)
    return {"pages": n_pages, "recorded": len(records), "counts": dict(sorted(counts.items())), "ok_empty": empty}


def compare(reference: dict[int, dict], other: dict[int, dict], index: list, n_pages: int) -> dict:
    mismatches = [i for i in range(n_pages) if page_key(reference[i]) != page_key(other[i])]
    return {
        "identical": n_pages - len(mismatches),
        "mismatched": len(mismatches),
        "examples": [
            {"i": i, "record_id": index[i].record_id, "reference": reference[i], "variant": other[i]}
            for i in mismatches[:MAX_EXAMPLES]
        ],
    }


def write_results(path: Path, records: dict[int, dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in sorted(records):
            f.write(json.dumps(records[i]) + "\n")


def read_results(path: Path) -> dict[int, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {record["i"]: record for record in map(json.loads, f)}


def write_golden(path: Path, index: list, records: dict[int, dict]) -> None:
    """``page, html_sha256[:16], status, text_sha256[:16] | error`` per page, gzip TSV, deterministic bytes."""
    lines = []
    for i, entry in enumerate(index):
        record = records[i]
        value = record.get("sha256", "")[:GOLDEN_HASH_CHARS] or record.get("error", "")
        lines.append(f"{i}\t{entry.html_sha256[:GOLDEN_HASH_CHARS]}\t{record['status']}\t{value}\n")
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        f.write("".join(lines).encode())


def parse_variants(args: argparse.Namespace) -> list[Variant]:
    pythons = dict(item.split("=", 1) for item in args.variant)
    package_dirs = dict(item.split("=", 1) for item in args.package_dir)
    preloads: dict[str, list[str]] = {}
    for item in args.preload:
        name, module = item.split("=", 1)
        preloads.setdefault(name, []).append(module)
    unknown = (set(package_dirs) | set(preloads)) - set(pythons)
    if unknown:
        raise ValueError(f"options for undeclared variants: {sorted(unknown)}")
    return [
        Variant(name, python, package_dirs.get(name), tuple(preloads.get(name, ()))) for name, python in pythons.items()
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--fixture", action="append", required=True, choices=sorted(FIXTURES))
    parser.add_argument("--variant", action="append", required=True, help="NAME=PYTHON")
    parser.add_argument("--package-dir", action="append", default=[], help="NAME=DIR (production layout)")
    parser.add_argument("--preload", action="append", default=[], help="NAME=MODULE imported before the extension")
    parser.add_argument("--reference", required=True, help="variant (or --reference-results run) others must equal")
    parser.add_argument("--reference-results", type=Path, help="directory of a reference run made elsewhere")
    parser.add_argument("--jobs", type=int, required=True, help="concurrent worker processes")
    parser.add_argument("--page-timeout", type=float, default=DEFAULT_PAGE_TIMEOUT)
    parser.add_argument("--write-golden", action="append", default=[], help="FIXTURE=PATH from the reference")
    args = parser.parse_args()

    variants = parse_variants(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"reference": args.reference, "variants": [v.__dict__ for v in variants], "fixtures": {}}
    goldens = dict(item.split("=", 1) for item in args.write_golden)
    passed = True
    for name in args.fixture:
        fixture = FIXTURES[name]
        warc = download(fixture, args.work_dir / f"{fixture.warc_hash}.warc.gz")
        pages, index_path = args.work_dir / f"pages_{name}.bin", args.work_dir / f"index_{name}.json"
        if not index_path.exists():
            write_pages(warc, pages, index_path)
        index = read_index(index_path)
        n_pages = len(index)
        logger.info("%s: %d HTML pages", name, n_pages)
        started = time.monotonic()
        results = run_variants(variants, name, pages, index_path, n_pages, args.work_dir, args.jobs, args.page_timeout)
        fixture_report: dict = {"warc": fixture.url, "pages": n_pages, "seconds": round(time.monotonic() - started)}
        if args.reference_results:
            with gzip.open(args.reference_results / f"index_{name}.json.gz", "rt") as f:
                reference_index = json.load(f)
            same_html = [e["html_sha256"] for e in reference_index] == [e.html_sha256 for e in index]
            fixture_report["same_html_as_reference_run"] = same_html
            passed &= same_html
            reference = read_results(args.reference_results / f"results_{name}_{args.reference}.jsonl.gz")
        else:
            reference = results[args.reference].records
        with gzip.open(args.work_dir / f"index_{name}.json.gz", "wt") as f:
            f.write(index_path.read_text())
        fixture_report["variants"] = {}
        for variant in variants:
            records = results[variant.name].records
            write_results(args.work_dir / f"results_{name}_{variant.name}.jsonl.gz", records)
            entry = {"summary": summarize(records, n_pages), "loaded": results[variant.name].headers[0]}
            if variant.name != args.reference or args.reference_results:
                entry["vs_reference"] = compare(reference, records, index, n_pages)
                passed &= entry["vs_reference"]["mismatched"] == 0
            fixture_report["variants"][variant.name] = entry
            logger.info("%s %s: %s", name, variant.name, json.dumps(entry.get("vs_reference", entry["summary"]))[:300])
        if name in goldens:
            write_golden(Path(goldens[name]), index, reference)
        report["fixtures"][name] = fixture_report
    report["passed"] = passed
    (args.work_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("parity %s; report at %s", "PASSED" if passed else "FAILED", args.work_dir / "report.json")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
