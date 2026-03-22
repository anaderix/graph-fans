"""Shared save utilities for plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, save_path: str | Path, title: str = "") -> None:
    """Save a figure as PNG and a companion .md file embedding it.

    Args:
        fig: Matplotlib figure.
        save_path: Base path (without extension). Will create .png and .md.
        title: Optional title for the markdown file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    png_path = save_path.with_suffix(".png")
    md_path = save_path.with_suffix(".md")

    fig.savefig(png_path, dpi=150, bbox_inches="tight")

    md_title = title or save_path.stem.replace("_", " ").replace("-", " ").title()
    md_content = f"# {md_title}\n\n![{md_title}]({png_path.name})\n"
    md_path.write_text(md_content)
