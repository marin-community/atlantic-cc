# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""R5 public decon parity: replay decontamination on published Atlantic-CC files and compare.

Uses only public inputs: release files from ``data.commoncrawl.org`` plus the shipped hash lists
(D in the package; S' and the S' name map as release assets). Every document must get exactly the
stored ``decon_matches`` (empty for ``part-*`` files, non-empty for ``contaminated-*``).

    uv run python -m tests.parity.decon.public_release --artifacts .cache/a5/artifacts --cache .cache/a5/release
"""

import argparse
import datetime
import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from atlantic_cc.decon.artifacts import (
    EVAL_HIT_NAMES_FILE,
    EVAL_HITS_FILE,
    load_drop_hashes,
    load_eval_hit_names,
    load_eval_hits,
)
from atlantic_cc.decon.filter import InputKind, decon_files, output_name
from atlantic_cc.release.params import HTTPS_ROOT, VERSION

logger = logging.getLogger(__name__)

RELEASE_URL = f"{HTTPS_ROOT}/{VERSION}"
REPORT = Path(__file__).parent / "reports" / "public_release.json"


@dataclass(frozen=True)
class Case:
    name: str
    files: tuple[str, ...]
    expected_flagged: dict[str, list[str]] | None = None
    """Exact flagged ids and names when given; otherwise every stored ``decon_matches`` must be reproduced."""


R1_CC_NEWS_2016_08 = Case(
    "R1 crawl=CC-NEWS-2016-08 (whole crawl)",
    (
        "crawl=CC-NEWS-2016-08/part-00000-of-00001.parquet",
        "crawl=CC-NEWS-2016-08/contaminated-00000-of-00001.parquet",
    ),
    {
        "f263c271-23da-49f3-8082-3d2bec260a34": ["paloma_4chan"],
        "3cdc9411-8745-4943-909a-200d7b9d3e5c": ["paloma_c4_100_domains"],
    },
)
R2_CONTAMINATED = Case(
    "R2 contaminated-only files",
    (
        "crawl=CC-NEWS-2024-05/contaminated-00000-of-00001.parquet",
        "crawl=CC-SUPPLEMENTAL-2026-22/contaminated-00000-of-00001.parquet",
    ),
)
CASES = (R1_CC_NEWS_2016_08, R2_CONTAMINATED)


def fetch(relative: str, cache: Path) -> Path:
    """Download a release file once into *cache* (same relative layout)."""
    local = cache / relative
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_suffix(local.suffix + ".partial")
        urllib.request.urlretrieve(f"{RELEASE_URL}/{relative}", partial)
        os.replace(partial, local)
    return local


def run_case(case: Case, cache: Path, out: Path, membership, drop, names, workers: int) -> dict:
    paths = [str(fetch(f, cache)) for f in case.files]
    t0 = time.time()
    summaries = decon_files(paths, InputKind.RELEASE, str(out), membership, drop, names, workers)
    flagged = {}
    for path in paths:
        table = pq.read_table(out / f"{output_name(path)}.decon.parquet")
        flagged.update(zip(table["id"].to_pylist(), table["decon_matches"].to_pylist(), strict=True))
    result = {
        "case": case.name,
        "files": list(case.files),
        "docs": sum(s["docs"] for s in summaries),
        "flagged": len(flagged),
        "stored_flagged": sum(s["stored_flagged"] for s in summaries),
        "mismatched": sum(s["mismatched"] for s in summaries),
        "mismatched_ids": [i for s in summaries for i in s["mismatched_ids"]][:50],
        "matched_hashes_without_names": sum(s["matched_hashes_without_names"] for s in summaries),
        "per_file": [{k: s[k] for k in ("path", "docs", "flagged", "stored_flagged", "mismatched")} for s in summaries],
        "seconds": round(time.time() - t0, 1),
    }
    for f in result["per_file"]:
        f["path"] = Path(f["path"]).relative_to(cache).as_posix()
    if case.expected_flagged is not None:
        result["expected_flagged"] = case.expected_flagged
        result["flagged_equal_expected"] = flagged == case.expected_flagged
    result["passed"] = result["mismatched"] == 0 and result.get("flagged_equal_expected", True)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--artifacts", type=Path, required=True, help=f"Directory with {EVAL_HITS_FILE} and {EVAL_HIT_NAMES_FILE}."
    )
    ap.add_argument("--cache", type=Path, required=True, help="Download cache for release files.")
    ap.add_argument(
        "--out", type=Path, default=None, help="Directory for per-file outputs (default: <cache>/decon_out)."
    )
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args()
    out = args.out or args.cache / "decon_out"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    membership = load_eval_hits(str(args.artifacts / EVAL_HITS_FILE))
    names = load_eval_hit_names(str(args.artifacts / EVAL_HIT_NAMES_FILE))
    drop = load_drop_hashes()
    load_seconds = round(time.time() - t0, 1)
    results = [run_case(c, args.cache, out, membership, drop, names, args.workers) for c in CASES]
    report = {
        "generated": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "membership": f"S' ({EVAL_HITS_FILE}, {len(membership)} hashes)",
        "drop_hashes": len(drop),
        "artifact_load_seconds": load_seconds,
        "cases": results,
        "passed": all(r["passed"] for r in results),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({r["case"]: (r["docs"], r["flagged"], r["mismatched"], r["passed"]) for r in results}, indent=1))


if __name__ == "__main__":
    main()
