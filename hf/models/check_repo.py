# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pre- and post-publication checks of an Atlantic-CC model repo (test M5 in ``docs/reproducibility.md``).

Checks a staged folder or a published Hub repo for:

- the two required README paragraphs, each exactly once;
- ``license: openmdw-1.1`` in the README YAML front matter;
- a root ``LICENSE`` byte-identical to the official OpenMDW-1.1 text;
- no unfilled template markers, no Hugging Face dataset links, and none of the terms listed in an
  optional ``--forbid-terms`` file (one term per line, matched case-insensitively);
- for a staged folder, every file listed in ``manifest.json`` with its size and SHA-256, and every
  relative link or image of the card (``cascade.png``, ``LICENSE``, ...) present.

Usage::

    uv run python hf/models/check_repo.py .cache/models/hf_staging/atlantic-cc-modernbert
    uv run python hf/models/check_repo.py marin-community/atlantic-cc-modernbert   # after upload
    uv run python hf/models/check_repo.py --forbid-terms terms.txt <folder or repo id> ...
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

RESEARCH_ARTIFACT = (
    "Research artifact. This model has not been properly tested or evaluated and is not necessarily secure. "
    "Use at your own risk in production settings."
)
RESPONSIBLE_USE = (
    "We release our tools to the public with the intent that they are used safely, ethically, and responsibly. "
    "They are meant to contribute to the public's understanding of science, and aid in its advancement. "
    "By using this Marin model, you agree to comply with all applicable law."
)
LICENSE_ID = "openmdw-1.1"
# sha256 of https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1 (also hf/models/LICENSE.OpenMDW-1.1)
OPENMDW_SHA256 = "2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df"
# Unfilled template markers; and the cards point to Common Crawl's distribution of the data rather than
# to a Hugging Face dataset repo.
FORBIDDEN = (
    "{{",
    "<!-- NUMERICS",
    "huggingface.co/datasets",
)


def _front_matter(readme: str) -> str:
    match = re.match(r"---\n(.*?)\n---\n", readme, flags=re.S)
    if match is None:
        raise ValueError("README.md has no YAML front matter")
    return match.group(1)


def problems(readme: str, license_bytes: bytes, extra_forbidden: tuple[str, ...] = ()) -> list[str]:
    """Everything wrong with a card and its LICENSE; empty when the repo may be published."""
    found = []
    for name, paragraph in (("research-artifact", RESEARCH_ARTIFACT), ("responsible-use", RESPONSIBLE_USE)):
        count = readme.count(paragraph)
        if count != 1:
            found.append(f"{name} paragraph appears {count} times (expected once)")
    if not re.search(rf"^license:\s*{re.escape(LICENSE_ID)}\s*$", _front_matter(readme), flags=re.M):
        found.append(f"README front matter lacks 'license: {LICENSE_ID}'")
    if hashlib.sha256(license_bytes).hexdigest() != OPENMDW_SHA256:
        found.append("LICENSE is not the official OpenMDW-1.1 text")
    found += [f"README contains {term!r}" for term in FORBIDDEN if term.lower() in readme.lower()]
    found += [f"README contains a forbidden term ({t!r})" for t in extra_forbidden if t.lower() in readme.lower()]
    return found


# Markdown link or image targets that are paths inside the repo (not URLs or anchors).
LOCAL_LINK = re.compile(r"\]\((?!https?://|#)([^)\s]+)\)")


def link_problems(directory: Path, readme: str) -> list[str]:
    """Relative links and images of the card (``cascade.png``, ``LICENSE``, ...) that name no file of the folder."""
    targets = sorted(set(LOCAL_LINK.findall(readme)))
    return [f"README links to {t!r}, which the folder lacks" for t in targets if not (directory / t).is_file()]


def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_problems(directory: Path) -> list[str]:
    manifest = json.loads((directory / "manifest.json").read_text())
    found = []
    listed = {entry["path"] for entry in manifest["files"]}
    on_disk = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()} - {"manifest.json"}
    if listed != on_disk:
        found.append(f"manifest lists {sorted(listed ^ on_disk)} inconsistently with the folder")
    for entry in manifest["files"]:
        path = directory / entry["path"]
        if path.exists() and (path.stat().st_size != entry["bytes"] or sha256_hex(path) != entry["sha256"]):
            found.append(f"{entry['path']} does not match manifest.json")
    return found


def check(source: str, extra_forbidden: tuple[str, ...] = ()) -> list[str]:
    """Problems of a local folder (including its manifest) or of a published Hub repo."""
    directory = Path(source)
    if directory.is_dir():
        readme, license_bytes = (directory / "README.md").read_text(), (directory / "LICENSE").read_bytes()
        return (
            problems(readme, license_bytes, extra_forbidden)
            + manifest_problems(directory)
            + link_problems(directory, readme)
        )
    readme = Path(hf_hub_download(source, "README.md")).read_text()
    license_bytes = Path(hf_hub_download(source, "LICENSE")).read_bytes()
    return problems(readme, license_bytes, extra_forbidden)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check staged folders or published Hub repos.")
    parser.add_argument("sources", nargs="+", help="staged folders or Hub repo ids")
    parser.add_argument("--forbid-terms", type=Path, help="file of extra terms the cards must not contain")
    args = parser.parse_args()
    lines = args.forbid_terms.read_text().splitlines() if args.forbid_terms else []
    extra = tuple(line.strip() for line in lines if line.strip())
    failed = False
    for source in args.sources:
        found = check(source, extra)
        failed |= bool(found)
        print(f"{source}: {'OK' if not found else 'FAIL'}")
        for problem in found:
            print(f"  - {problem}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
