# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract pages ``[start, end)`` of a pages file with the ``resiliparse._extract_rs`` this interpreter imports.

Standard library only: it runs under each extractor's own interpreter (``python -I worker.py ...``). The
first output line describes the loaded extension; then one JSON line per page, flushed as written, so a
supervisor can tell which page killed the process. Each page gets production's call,
``extract_plain_text(html, main_content=True, preserve_formatting="markdown")``, without production's
screens: crashes and exceptions are part of what is compared.
"""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


def sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pages")
    parser.add_argument("index")
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("out")
    parser.add_argument("--package-dir", help="put this directory first on sys.path (production layout)")
    parser.add_argument("--preload", action="append", default=[], help="import this module first")
    parser.add_argument("--dump-dir", help="also write each page's text to DIR/<page>.txt")
    args = parser.parse_args()

    if args.package_dir:
        sys.path.insert(0, args.package_dir)
    for module in args.preload:
        importlib.import_module(module)
    extension = importlib.import_module("resiliparse._extract_rs")
    extract_plain_text = extension.extract_plain_text
    lexbor = sorted(p for p in Path(extension.__file__).parent.iterdir() if p.name.startswith("liblexbor"))

    index = json.loads(Path(args.index).read_text())
    with open(args.pages, "rb") as pages, open(args.out, "a", encoding="utf-8") as out:
        header = {
            "header": True,
            "extension": extension.__file__,
            "extension_sha256": sha256_file(extension.__file__),
            "lexbor": {p.name: sha256_file(str(p)) for p in lexbor},
            "preloaded": sorted(m for m in sys.modules if m.startswith("resiliparse")),
            "python": sys.version.split()[0],
        }
        out.write(json.dumps(header) + "\n")
        out.flush()
        for i in range(args.start, args.end):
            pages.seek(index[i]["offset"])
            html = pages.read(index[i]["length"]).decode("utf-8", "surrogatepass")
            try:
                text = extract_plain_text(html, main_content=True, preserve_formatting="markdown")
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:  # pyo3's PanicException derives from BaseException
                record = {"i": i, "status": "exception", "error": type(e).__name__}
            else:
                data = text.encode("utf-8")
                record = {"i": i, "status": "ok", "sha256": hashlib.sha256(data).hexdigest(), "chars": len(text)}
                if args.dump_dir:
                    Path(args.dump_dir, f"{i}.txt").write_bytes(data)
            out.write(json.dumps(record) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
