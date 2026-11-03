# The resiliparse-rs extractor

The run that built Atlantic-CC v1.0 ("production" below) took the text of every page it extracted
from `resiliparse._extract_rs`, the Rust main-content extractor of the resiliparse-rs fork
[`marin-community/atlantic-cc-resiliparse`](https://github.com/marin-community/atlantic-cc-resiliparse)
(tag [`atlantic-cc-v1.0`](https://github.com/marin-community/atlantic-cc-resiliparse/tree/atlantic-cc-v1.0))
of [chatnoir-eu/chatnoir-resiliparse](https://github.com/chatnoir-eu/chatnoir-resiliparse), at commit
[`850891b`](https://github.com/marin-community/atlantic-cc-resiliparse/tree/850891b277e919f671ec19f12aafdce451b327f4).
Production built its binaries in August 2026 from the same commit, then hosted at
`XenonMolecule/chatnoir-resiliparse`. Production ran on Marin's compute cluster, whose job scheduler
is called Iris; both names appear below. The fork is not on PyPI. There are two supported ways to
install it:

- **As-run wheel.** The exact binaries production loaded, repackaged byte for byte. Linux x86_64,
  glibc >= 2.34, CPython 3.12.
- **Source build.** `scripts/build_resiliparse_rs/` builds the same commit with pinned toolchain
  inputs. The Docker build targets manylinux_2_28 (glibc >= 2.28); `build.sh` also runs natively on
  Linux x86_64 and macOS arm64.

Two Common Crawl WARC files serve as test fixtures: F1 (CC-MAIN-2014-23,
`crawl-data/CC-MAIN-2014-23/segments/1405997884573.18/warc/CC-MAIN-20140722025804-00181-ip-10-33-131-23.ec2.internal.warc.gz`)
and F3 (CC-MAIN-2026-12,
`crawl-data/CC-MAIN-2026-12/segments/1772687277590.81/warc/CC-MAIN-20260308022436-20260308052436-00349.warc.gz`),
both under `https://data.commoncrawl.org/`. On their 32,628 HTML pages, every Linux build and install
layout we tested gave the same result as the production binaries on every page: the same status
(text or crash) and byte-identical text ([parity results](#parity-results)). Built in production's
environment, the source build reproduces the production binaries themselves: both `.so` files have
the production md5s.

## The extractor

The fork adds a Rust port of resiliparse's `extract_plain_text` (crate `resiliparse` in
`resiliparse-rs/`) with main-content heuristics, a gradient-boosted block classifier compiled into
the binary (120 trees of depth 6), and a Markdown output mode. The pyo3 crate
`resiliparse-extract-rs` (`resiliparse-py/resiliparse/extract/_extract_rs/`) exposes it as the Python
module `_extract_rs`. HTML is parsed with lexbor (phoerious/lexbor @ 647a0b5), linked as a shared
library.

Production called, for every page that passed its size and frameset screens
([crash and hang behavior](#crash-and-hang-behavior)), with `html` the page decoded to a `str`:

```python
from resiliparse._extract_rs import extract_plain_text

text = extract_plain_text(html, main_content=True, preserve_formatting="markdown")
```

The other keyword arguments keep the binding's defaults: `list_bullets=True, alt_texts=True,
links=False, form_fields=False, noscript=False, comments=True, post_meta=True,
hidden_elements=False, skip_elements=None`. The call releases the GIL while it extracts. For a given
build and math library, output is a pure function of the input string: there is no randomness, and
the extractor's hash-map uses do not depend on iteration order.

`resiliparse.extract.html2text` is upstream's engine and gives different text.

## Pinned build inputs

| input | pin |
|---|---|
| fork | `marin-community/atlantic-cc-resiliparse` @ `850891b277e919f671ec19f12aafdce451b327f4` (tag `atlantic-cc-v1.0`) |
| Rust | 1.98.0 (`88d9e12ae 2026-08-18`); crates locked by the fork's `Cargo.lock` (pyo3 0.28.3, regex 1.13.1, parking_lot 0.12.5, bindgen 0.72.1) |
| CPython | 3.12. pyo3 builds here are not abi3, so a build loads only in the CPython minor it was built for. |
| lexbor | fork overlay port `.vcpkg/ports/lexbor`: `phoerious/lexbor` @ `647a0b5cd5f35e1e575c20194595dc449abd0e7b` (archive SHA512 `a8d7c71a197edab5446bf757bfda4653cb839d69c59e0e6dc80ddf5512c94103e3c54c6a52ffc4de882dea3c7901b1103e7b5c835b80403feee48c0c9ef48d83`), shared library, `-DLEXBOR_OPTIMIZATION_LEVEL=-O3`, installs `liblexbor.so.2.4.0` |
| vcpkg ports | registry baseline `71f123418c6cdc00258964629b0ede4317ba1a36` (fork's `vcpkg-configuration.json`); overlay triplets `.vcpkg/triplets` (`x64-linux`: dynamic, release-only, `-O3`) |
| vcpkg checkout and tool | `microsoft/vcpkg` @ `dc31f86a441067d6c2e9e4ad52988eb41ef4f442` (master during production's lexbor build, 2026-08-11); tool release `2026-07-27`, SHA512-checked |
| cmake, ninja, libclang | pip `cmake==4.4.2`, `ninja==1.13.0`, `libclang==18.1.1` (Linux) |
| C compiler | production: `gcc (Debian 14.2.0-19) 14.2.0`; Docker build: gcc-toolset-14 (`14.2.1 20250110`) |
| `vcpkg.json` | dependencies trimmed to `["lexbor"]` (the other entries are fastwarc dependencies) |
| `RUSTFLAGS` | `-C link-arg=-Wl,-rpath,$ORIGIN` |

The rustc and gcc versions come from the `.comment` sections of the production binaries. Setting
`RUSTFLAGS` replaces `build.rustflags = ["--cfg", "pyo3_disable_reference_pool"]` from the fork's
`.cargo/config.toml`, so production was built without that cfg. The cfg changes how pyo3 defers
reference-count updates made without the GIL, which should not change extracted text; we did not
test a build with the cfg on. `build.sh` sets `RUSTFLAGS` as production did. The fork's
`INSTALL_RUST_EXTRACTOR.md` does not, so a build that follows it has the cfg on.

## Option 1: the as-run wheel

`resiliparse_rs_atlantic_cc-0+850891b-cp312-cp312-manylinux_2_34_x86_64.whl`
(sha256 `7747e6e4dce15057df3082275fcdd722a4a81a571aeda33dc919365baf1439f8`, 3.2 MB) contains
production's two binaries, unchanged:

| file in the wheel | bytes | md5 (base64) | sha256 |
|---|---|---|---|
| `resiliparse/_extract_rs.cpython-312-x86_64-linux-gnu.so` | 3,979,816 | `mjHiDXINNTxVCUttKER1Iw==` | `9e2bdfe91dc14fbe17846c0d1c0806eb12b19f64bba6db9cf8669fd1bd19a364` |
| `resiliparse/liblexbor.so.2` | 12,431,649 | `hX51w84LHTU6X/dTPVUyfw==` | `cf016d48cf67a8b6d859cb25e71c82c3c7ff414cfcaedf595e8b3c158f5aecd2` |

The wheel is not published yet. Until it is, it exists only where maintainers build it (see the end of
this section). Install it into the project environment and smoke-test it:

```bash
uv pip install resiliparse_rs_atlantic_cc-0+850891b-cp312-cp312-manylinux_2_34_x86_64.whl
uv run atlantic-cc extractor-check     # prints "smoke": "passed" and "asrun_binaries": true
```

The wheel is not in the project's lockfile, and `uv sync` removes every package the lockfile does not
list. Run `uv sync` first, with every extra you need (for example `--extra torch --extra dev`), and
install the wheel after it, or sync with `uv sync --inexact`. `uv run` does not remove it.

The highest symbol versions the binaries need are `GLIBC_2.34` (`pthread_key_create`), `GLIBC_2.33`
and `GLIBC_2.29` (`log`), so the wheel installs on glibc 2.34 or newer: RHEL, Alma and Rocky Linux 9,
Ubuntu 22.04, Debian 12, Amazon Linux 2023 and later. `auditwheel show` agrees with the
`manylinux_2_34_x86_64` tag. On older systems use the Docker build.

Production built them in an Iris CPU task (Debian 13, glibc 2.41): `liblexbor.so.2` on 2026-08-11 and
`_extract_rs.so` on 2026-08-26, and stored them in Marin's internal buckets
(`gs://marin-<region>/artifacts/resiliparse_rs/latest/`, one copy per cloud region, all identical).
The extension's RUNPATH is
`$ORIGIN:/root/resiliparse_rs_build/chatnoir-resiliparse/vcpkg_installed/x64-linux/lib`. The second
entry is the build tree and does not exist on other machines; lexbor resolves through `$ORIGIN`.

Maintainers with access to those buckets rebuild the wheel from the production artifact with
`scripts/build_resiliparse_rs/package_asrun.sh ARTIFACT_DIR OUT_DIR`. It checks both md5s against
`atlantic_cc.spec.ASRUN_EXTRACTOR_ARTIFACT` and writes a byte-identical wheel on every run.

## Option 2: build from source

### Docker (portable, glibc >= 2.28)

From the Atlantic-CC repository root, on any machine with Docker (install the wheel, as in Option 1,
on the Linux x86_64 machine that will run extraction):

```bash
docker build --platform linux/amd64 -t atlantic-cc-resiliparse-rs scripts/build_resiliparse_rs
docker run --rm --platform linux/amd64 -v "$PWD/dist:/out" atlantic-cc-resiliparse-rs
uv pip install dist/resiliparse_rs_atlantic_cc-0+850891b-cp312-cp312-manylinux_2_28_x86_64.whl
```

The image is `quay.io/pypa/manylinux_2_28_x86_64:2026.10.26-1` (AlmaLinux 8, glibc 2.28), pinned by
digest, plus rustup 1.29.0 (checksum-verified) and Rust 1.98.0. The build downloads the fork, vcpkg,
lexbor and the Rust crates, so it needs network access. It took about 3 minutes on 8 vCPUs; under
x86_64 emulation on an ARM host expect much longer. The wheel's binaries need at most `GLIBC_2.28` and
also link `libpthread.so.0`, as glibc before 2.34 requires. They differ from the production binaries
byte-wise and gave identical text on every fixture page.

### Native build

```bash
scripts/build_resiliparse_rs/build.sh --out-dir dist --python python3.12 [--work-dir DIR]
```

Supported build hosts are Linux x86_64 (glibc) and macOS arm64. `--work-dir` defaults to
`./.cache/resiliparse_rs_build`. The script needs git, curl, tar, a C compiler (gcc on Linux, the
Xcode command line tools on macOS), CPython 3.12 and rustup; it installs Rust 1.98.0 through rustup.
cmake, ninja and (on Linux) libclang are pip-installed into a venv under the work directory. The
script:

1. fetches the fork at `850891b` and vcpkg at `dc31f86`, and SHA512-checks the vcpkg tool;
2. trims `vcpkg.json` to lexbor and runs `vcpkg install --triplet x64-linux` (`arm64-osx` on macOS);
3. runs `cargo build --release --locked -p resiliparse-extract-rs` with production's `RUSTFLAGS`
   (`@loader_path` on macOS);
4. runs the fork's golden tests, `cargo test --release --locked -p resiliparse --test extract_golden`
   (7 tests: the 5 golden pages and 2 smaller checks);
5. writes to `--out-dir` the wheel (`scripts/build_resiliparse_rs/package_wheel.py`; platform tag
   from the binaries' highest `GLIBC_2.x` requirement), a production-layout `artifact/` directory,
   `atlantic_cc_build.json`, `golden_tests.log` and `contract_check.json` (the result of step 6);
6. installs the wheel into a fresh venv, hides the build tree, and checks that the extension imports
   and reproduces the fork's 5 golden pages exactly.

Downloads and build files stay under the work directory, except that rustup and cargo use the
caller's `RUSTUP_HOME` and `CARGO_HOME` (default `~/.rustup` and `~/.cargo`). vcpkg binary caching is
off.

To reproduce the production binaries byte for byte, build in production's environment: Iris's
default task image (`ghcr.io/marin-community/iris-task`, which is `python:3.12-slim` plus
`build-essential` and rustup; Debian 13, `gcc (Debian 14.2.0-19) 14.2.0`), work directory
`/root/resiliparse_rs_build` and `CARGO_HOME=/cargo`. `tests/parity/extractor/cluster_build.sh debian`
does this as an Iris job; on 2026-11-02 it produced `_extract_rs.so` and `liblexbor.so.2` with the md5s
in the table above. Another Debian 13 container with the same gcc package, paths and Rust version
should give the same bytes, but we only ran the Iris image, and production's image is not pinned by
digest. Rust embeds dependency source paths in the binary, so a different work directory or
`CARGO_HOME` changes the bytes, though not the extracted text.

On macOS arm64 the build gives `_extract_rs.cpython-312-darwin.so` plus `liblexbor.2.4.0.dylib` in a
`macosx_11_0_arm64` wheel. It passes the golden tests. See [parity results](#parity-results) for how
its output compares with production's.

## Wheel layout

Both wheel kinds install the same layout, distribution `resiliparse-rs-atlantic-cc`, version
`0+850891b`:

```
resiliparse/_extract_rs.cpython-312-x86_64-linux-gnu.so   (macOS: _extract_rs.cpython-312-darwin.so)
resiliparse/liblexbor.so.2                               (macOS: liblexbor.2.4.0.dylib)
resiliparse_rs_atlantic_cc-0+850891b.dist-info/atlantic_cc_build.json
resiliparse_rs_atlantic_cc-0+850891b.dist-info/licenses/{LICENSE, lexbor/LICENSE, lexbor/NOTICE, THIRD_PARTY_RUST_CRATES.txt}
```

The extension loads lexbor from its own directory through the `$ORIGIN` (`@loader_path`) rpath, the
layout production used. The wheel leaves out the rest of the fork's `resiliparse-py` tree, which
production unpacked next to the binaries: on production's import path only its
`resiliparse/__init__.py` ran, and that file holds only a license header. The other files are
upstream's Cython sources, which production never compiled or imported.

The wheel ships no `resiliparse/__init__.py`. Without upstream `resiliparse` installed, `resiliparse`
is a namespace package. With it, the two files sit inside upstream's package directory and neither
distribution owns a file of the other. We checked this in fresh venvs with upstream `resiliparse`
1.0.9 (Cython) and 2.0.0 (Rust), installing in both orders. Both packages imported and worked side by
side. Uninstalling either one left the other working. Text was identical with upstream's native
modules loaded in the same process. Coexistence needs both distributions in the same site-packages
directory: a regular `resiliparse` package earlier on `sys.path` hides this one.

`atlantic_cc_build.json` records the kind (`as-run` or `source`), the fork commit, the platform tag,
each file's size, sha256 and md5, and the toolchain:

```python
import importlib.metadata, json

print(json.loads(importlib.metadata.distribution("resiliparse-rs-atlantic-cc").read_text("atlantic_cc_build.json")))
```

`atlantic_cc.extract.pool.open_extractor_pool(n_procs, pkg_dir=None)` uses an installed wheel
directly. `atlantic_cc.extract.install.install_artifact` accepts production's artifact layout
(`manifest.json`, `resiliparse_py.tar.gz`, `_extract_rs.so`, `liblexbor.so.2`), which `build.sh`
writes to `OUT_DIR/artifact/`; pass the directory it returns as `pkg_dir`.

## Verifying an install

1. Smoke test. `uv run atlantic-cc extractor-check` (about a second) extracts a small page in a spawned
   process, as every phase-A process does at startup, and reports `asrun_binaries`: whether the
   extension that loads is production's build. With an artifact directory, add `--pkg-dir DIR`.

2. Hashes. For the as-run wheel, compare both installed files with the table above:

   ```bash
   uv run python -c "
   import hashlib, pathlib, resiliparse._extract_rs as m
   for p in (pathlib.Path(m.__file__), pathlib.Path(m.__file__).with_name('liblexbor.so.2')):
       print(p, hashlib.sha256(p.read_bytes()).hexdigest())"
   ```

3. F1 golden. `tests/parity/extractor/test_extractor_golden.py` downloads F1 (222 MB) from
   data.commoncrawl.org, extracts its 13,516 HTML pages with the installed extension in
   crash-isolated worker processes, and compares each result with
   `tests/parity/extractor/golden_f1_asrun.tsv.gz`: the production binaries' status and the first 16
   hex digits of the text's sha256, per page. It needs about 1.2 GB of scratch space; on an 8-core
   laptop it took 3 minutes, including the download.

   ```bash
   uv run --extra dev pytest -m "network and extractor" tests/parity/extractor/test_extractor_golden.py
   ```

   To test a wheel without installing it, add `--with <wheel>` after `--extra dev`. A skip means the
   test could not import `resiliparse._extract_rs`; it is not a pass.

4. Full parity. `tests/parity/extractor/run_parity.py` compares any number of installs ("variants")
   page by page on F1 and F3. Each page runs in a supervised worker process, so a segfault or hang
   costs one page and is recorded as `crash` or `timeout`. A variant passes when its status and text
   sha256 equal the reference's on every page. `cluster_build.sh` and `cluster_parity.sh` in the same
   directory are the Iris job bodies used for the results below.

## Parity results

Run on 2026-11-02 as two Iris jobs (8 CPUs each; the second with the final scripts and builds, same
results). The reference is the as-run wheel
in a fresh venv. `tests/parity/extractor/pages.py` selects records as production did and decodes them
with a simple deterministic rule, not production's charset cascade (`atlantic_cc.warc`): parity only
needs every variant to receive the same string. The page counts equal production's per-WARC count of
HTML records (`a_n_in`) for both WARCs.

| variant | binaries | F1 identical | F3 identical |
|---|---|---|---|
| `asrun` (reference) | production | 13,516 / 13,516 | 19,112 / 19,112 |
| `prod-layout` | production artifact unpacked and put first on `sys.path`, as production's workers loaded it | 13,516 / 13,516 | 19,112 / 19,112 |
| `src-debian` | source build, production environment (byte-identical binaries) | 13,516 / 13,516 | 19,112 / 19,112 |
| `src-manylinux` | source build, manylinux_2_28 (different binaries) | 13,516 / 13,516 | 19,112 / 19,112 |
| `asrun-up109` | as-run wheel with resiliparse 1.0.9's Cython modules loaded first | 13,516 / 13,516 | 19,112 / 19,112 |
| `asrun-up200` | as-run wheel with resiliparse 2.0.0's native module loaded first | 13,516 / 13,516 | 19,112 / 19,112 |

Results per page, identical for every variant: F1 has 13,477 `ok` (232 of them empty) and 39 `crash`;
F3 has 19,109 `ok` (643 empty) and 3 `crash`. Raised exceptions: none.

The macOS arm64 source build (Apple clang 17, rustc 1.98.0) was run on a laptop on the same decoded
HTML (checked by HTML sha256) and compared with the Linux results above: identical status and text on
all 13,516 F1 pages and all 19,112 F3 pages, including the same 42 crashes (39 in F1, 3 in F3). This
is evidence for macOS, not a guarantee: the block classifier calls `ln()` from the platform's libm, and
Atlantic-CC promises exact reproduction of the CPU stages on Linux x86_64 only
([reproducibility.md](reproducibility.md)).

## Crash and hang behavior

The extension can kill its process, and production has no per-document timeout. A reimplementation
must handle both the way production did, or its output differs from the release.

- **Segfaults.** Every crash in the parity run (39 in F1, 3 in F3) is a page that contains
  `<frameset` or `<FRAMESET`. Not every frameset page crashes: 42 F1 pages and 3 F3 pages contain
  `<frameset` in some letter case, and 39 and 3 of them crash. Production returns `""` without calling
  the extension when the HTML contains `<frameset` or `<FRAMESET`. The check is case-sensitive, so a
  mixed-case `<FrameSet` still reaches the engine. Production ran extraction in a crash-isolating
  process pool (ported as `atlantic_cc.extract.pool`): a crashed page yields `""` and is counted in
  the per-WARC crash counter `a_n_extract_crashed`. After the screen, production recorded 31 crashes in
  70.3M documents; its records attribute most of them to pages Common Crawl truncated at 1 MiB.
- **Exceptions.** A Rust panic surfaces as pyo3's `PanicException`, which derives from
  `BaseException`. A `str` holding lone surrogates raises `UnicodeEncodeError` before extraction starts.
  Production catches every `BaseException` except `KeyboardInterrupt` and `SystemExit` and returns `""`.
- **Size cap.** Production skips pages longer than 50,000,000 code points (`""`).
- **Hangs.** There is no per-document timeout. A page that hangs the engine stalls its WARC until
  production's 25-minute stall watchdog kills the worker; after 3 failed attempts a WARC was recorded
  as abandoned with zero documents. 101 of the run's 8,582,115 WARCs ended this way, from hangs or
  other repeated worker failures. Adding a timeout would produce documents the release lacks.
- **Loading.** The extension must be loaded from a filesystem that allows executable mappings; it
  fails to load from a `noexec` mount, such as `/tmp` on Iris workers.

## Licensing

| component | license | shipped in the wheel's `dist-info/licenses/` |
|---|---|---|
| fork (resiliparse crates and Python binding) | Apache-2.0 | `LICENSE` |
| lexbor (phoerious/lexbor @ 647a0b5) | Apache-2.0, with a NOTICE ("Lexbor. Copyright 2018-2020 Alexander Borisov") | `lexbor/LICENSE`, `lexbor/NOTICE` |
| Rust crates linked into the extension (pyo3, regex, parking_lot, memchr, …) | `MIT OR Apache-2.0`, except memchr and aho-corasick: `Unlicense OR MIT` | `THIRD_PARTY_RUST_CRATES.txt` (from `cargo tree`) |
| Rust standard library | `MIT OR Apache-2.0` | listed in `THIRD_PARTY_RUST_CRATES.txt` |

Every component allows redistribution of the binaries under Apache-2.0, or under the Unlicense for
the two crates that do not offer Apache-2.0.
