# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the classifier-cascade figure of each model card: ``hf/models/<repo>/cascade.png``.

A left-to-right chain from Common Crawl pages through extraction and the three classifiers to
Atlantic-CC (after deduplication and decontamination), with the card's own model highlighted. The PNGs
are committed next to the card templates and copied into the staged repos by ``stage.py``.

Usage::

    uv run --with matplotlib python hf/models/cascade_figure.py
"""

from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from atlantic_cc import spec

HF_DIR = Path(__file__).resolve().parent
WIDTH, HEIGHT, DPI = 8.0, 1.3, 200  # inches at 200 dpi: 1600 x 260 px
ACCENT, ACCENT_TEXT = "#2b6cb0", "white"
GREY_FILL, GREY_EDGE, GREY_TEXT = "#f1f3f5", "#adb5bd", "#495057"
ARROW = "#868e96"
BOX_W, BOX_H, BOX_Y = 1.0, 0.46, 0.74
GAP = 0.06  # between an arrow and a box
# (repo or None for a processing step, box label, caption), left to right
STEPS = (
    (None, "resiliparse-rs", "main-content text"),
    (spec.FASTTEXT_HF_REPO, "fastText", "full text, every page"),
    (spec.POOLED_HF_REPO, "pooled\ntransformer", "8,192 tokens"),
    (spec.TERMINAL_HF_REPO, "ModernBERT", "first 2,048 tokens, band only"),
)
BOX_X = (1.85, 3.15, 4.45, 5.75)  # box centers
SOURCE_X, SINK_X = 0.55, 7.55


def arrow(ax, x0: float, x1: float) -> None:
    ax.annotate("", xy=(x1, BOX_Y), xytext=(x0, BOX_Y), arrowprops={"arrowstyle": "-|>", "color": ARROW, "lw": 1.2})


def box(ax, x: float, label: str, *, step: bool, mine: bool) -> None:
    """A rounded box: dashed and white for a processing step, accent blue for this repo's model, else grey."""
    ax.add_patch(
        FancyBboxPatch(
            (x - BOX_W / 2, BOX_Y - BOX_H / 2),
            BOX_W,
            BOX_H,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="white" if step else ACCENT if mine else GREY_FILL,
            edgecolor=ACCENT if mine else GREY_EDGE,
            linestyle="--" if step else "-",
            linewidth=1.2,
        )
    )
    color, weight = (ACCENT_TEXT, "bold") if mine else (GREY_TEXT, "normal")
    ax.text(x, BOX_Y, label, ha="center", va="center", fontsize=8.5, color=color, weight=weight)


def render(highlight: str, out: Path) -> None:
    fig = Figure(figsize=(WIDTH, HEIGHT), dpi=DPI)  # rendered with Agg; no pyplot state
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    ax.text(SOURCE_X, BOX_Y, "Common Crawl\nHTML pages", ha="center", va="center", fontsize=8, color=GREY_TEXT)
    ax.text(SINK_X, BOX_Y, "Atlantic-CC", ha="center", va="center", fontsize=9, color=GREY_TEXT, weight="bold")
    for (repo, label, caption), x in zip(STEPS, BOX_X, strict=True):
        box(ax, x, label, step=repo is None, mine=repo == highlight)
        ax.text(x, BOX_Y - BOX_H / 2 - 0.16, caption, ha="center", va="center", fontsize=6.5, color=GREY_TEXT)
    arrow(ax, SOURCE_X + 0.5, BOX_X[0] - BOX_W / 2 - GAP)
    for left, right in zip(BOX_X, BOX_X[1:], strict=False):
        arrow(ax, left + BOX_W / 2 + GAP, right - BOX_W / 2 - GAP)
    last_x0, last_x1 = BOX_X[-1] + BOX_W / 2 + GAP, SINK_X - 0.44
    arrow(ax, last_x0, last_x1)
    ax.text(
        (last_x0 + last_x1) / 2, BOX_Y + 0.12, "dedup + decon", ha="center", va="bottom", fontsize=6, color=GREY_TEXT
    )
    fig.savefig(out, dpi=DPI, facecolor="white", metadata={"Software": None}, pil_kwargs={"optimize": True})


def main() -> None:
    for repo, _, _ in STEPS[1:]:
        out = HF_DIR / repo.split("/")[1] / "cascade.png"
        render(repo, out)
        print(f"wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
