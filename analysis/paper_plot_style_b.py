"""Shared paper typography and physically fixed plotting panels for the extended b.ipynb; sizes and panel geometry match a.ipynb."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PANEL_INCHES = 4.5
FONT_SIZE = 24
TICK_SIZE = 22
TITLE_SIZE = 26
ANNOTATION_SIZE = 21
INK = "#111827"
MUTED = "#687985"
PREFERRED = "#F5F8FC"
MODEL_MARKERS = {"AR": "o", "LL": "s", "Head": "D", "Adapt": "^",
                 "Laya-Med": "v", "NAR": "v", "Jev": "*", "MedJev": "+", "Laya": "x"}
MODEL_COLORS = {"AR": "#4C78A8", "LL": "#F2A65A", "Head": "#8C564B",
                "Adapt": "#D979B5", "Laya-Med": "#6B7280", "NAR": "#6B7280",
                "Jev": "#59A14F", "MedJev": "#D65F5F", "Laya": "#8C78B8"}
MODEL_SIZES = {"AR": 16, "LL": 16, "Head": 20, "Adapt": 16,
               "Laya-Med": 20, "NAR": 20, "Jev": 27, "MedJev": 18, "Laya": 18}
MODEL_DASHES = {"AR": (0, (4, 2)), "LL": "-", "Head": "-",
                "Adapt": (0, (7, 2)), "Laya-Med": (0, (5, 2, 1.5, 2))}


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE, "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold", "axes.labelpad": 12,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE, "legend.title_fontsize": FONT_SIZE,
        "axes.labelcolor": INK, "text.color": INK, "axes.titlecolor": INK,
        "axes.edgecolor": "#485463", "axes.linewidth": 1.1,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.major.pad": 10, "ytick.major.pad": 10,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": "#E5E9EF", "grid.linewidth": .7, "grid.alpha": .8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 2.0,
        "legend.columnspacing": 1.5, "legend.labelspacing": 0.6,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "lines.linewidth": 2.8, "lines.markersize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 220,
    })


def panel_grid(nrows, ncols, legend=False, sharex=False, sharey=False, xgap=None,
               ygap=2.0, right=1.6, top=0.85):
    """Reserve margins separately: every actual plotting box is 4.5 x 4.5 in."""
    left = 2.2
    bottom = 2.5 if legend else 1.25
    xgap = (0.8 if sharey else 1.25) if xgap is None else xgap
    width = left + right + ncols * PANEL_INCHES + (ncols - 1) * xgap
    height = bottom + top + nrows * PANEL_INCHES + (nrows - 1) * ygap
    fig = plt.figure(figsize=(width, height), layout=None)
    axes = np.empty((nrows, ncols), dtype=object)
    for row in range(nrows):
        for col in range(ncols):
            x = left + col * (PANEL_INCHES + xgap)
            y = bottom + (nrows - row - 1) * (PANEL_INCHES + ygap)
            ax = fig.add_axes([x / width, y / height, PANEL_INCHES / width, PANEL_INCHES / height],
                              sharex=axes[0, 0] if sharex and (row or col) else None,
                              sharey=axes[0, 0] if sharey and (row or col) else None)
            ax.set_box_aspect(1)
            if sharey and col:
                ax.tick_params(labelleft=False)
            axes[row, col] = ax
    fig._paper_axes = axes.ravel().tolist()
    return fig, np.squeeze(axes)


def panel_title(ax, index, text):
    ax.set_title(f"({chr(97 + index)})  {text}", loc="left", pad=18)


def prefer(ax, horizontal="left", vertical="top"):
    ax.add_patch(Rectangle((0 if horizontal == "left" else .5, .5 if vertical == "top" else 0),
                           .5, .5, transform=ax.transAxes, facecolor=PREFERRED,
                           edgecolor="none", zorder=0))


def legend_panel(ax, handles, labels, title=None):
    ax.axis("off")
    ax.legend(handles, labels, loc="center", title=title, alignment="left",
              borderaxespad=0, handlelength=1.6)


def figure_legend(fig, handles, labels, ncols=4, row=0):
    legend = fig.legend(handles, labels, loc="upper center", ncols=ncols,
                        bbox_to_anchor=(.5, .15), borderaxespad=0,
                        handlelength=1.8, columnspacing=1.2)
    if not hasattr(fig, "_paper_legends"):
        fig._paper_legends = []
    fig._paper_legends.append((row, legend))
    return legend


def model_marker(model, hollow=False, scale=1.0):
    """Pareto-compatible marker shapes, optical sizes, colors, and outlines."""
    color = MODEL_COLORS[model]
    outlined = model in ("Head", "Jev", "Laya-Med", "NAR")
    return dict(marker=MODEL_MARKERS[model], markersize=(22 if hollow else MODEL_SIZES[model]) * scale,
                markerfacecolor="none" if hollow else color,
                markeredgecolor=color if hollow or not outlined else INK,
                markeredgewidth=2.3 if hollow or outlined else (3.2 if model in ("Laya", "MedJev") else .9))


def model_handle(model, label=None, line=False):
    from matplotlib.lines import Line2D
    return Line2D([], [], color=MODEL_COLORS[model], linewidth=3.0,
                  linestyle=MODEL_DASHES.get(model, "-") if line else "none",
                  label=label or model, **model_marker(model, hollow=model == "AR", scale=.7 if line else 1))


def enlarge_coverage_axes(axes):
    for ax in np.ravel(axes):
        ax.tick_params(axis="both", labelsize=27, pad=9)
        ax.xaxis.label.set_size(28)
        ax.yaxis.label.set_size(28)


def risk_limits(values):
    """A shared zero-based risk range that fits every observed point."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    upper = min(1.02, max(.2, np.ceil((finite.max() + .015) * 10) / 10))
    return 0, upper


def add_colorbar(fig, ax, image, label):
    # A separate axes preserves the heatmap's physical plotting area.
    box = ax.get_position()
    cax = fig.add_axes([box.x1 + .16 / fig.get_figwidth(), box.y0 + box.height * .08,
                       .16 / fig.get_figwidth(), box.height * .84])
    bar = fig.colorbar(image, cax=cax, label=label)
    bar.outline.set_visible(False)
    bar.ax.tick_params(size=0, labelsize=18, pad=8)
    bar.ax.yaxis.label.set_size(20)
    return bar


def save_figure(fig, directory, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if hasattr(fig, "_paper_legends"):
        # Anchor the legend just below the lowest axis label, not at the canvas bottom.
        top = min(ax.get_tightbbox(renderer).y0 for ax in fig._paper_axes) - .24 * fig.dpi
        for _, legend in sorted(fig._paper_legends, key=lambda item: item[0], reverse=True):
            legend.set_bbox_to_anchor((.5, top / fig.bbox.height), transform=fig.transFigure)
            fig.canvas.draw()
            top = legend.get_window_extent(renderer).y0 - .14 * fig.dpi
        for legend in fig.legends:
            legend_box = legend.get_window_extent(renderer)
            if any(legend_box.overlaps(ax.get_tightbbox(renderer)) for ax in fig._paper_axes):
                raise ValueError(f"{stem}: legend overlaps a subplot or its labels")
    for ax in fig._paper_axes:
        box = ax.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        if not np.allclose([box.width, box.height], [PANEL_INCHES, PANEL_INCHES], atol=.001):
            raise ValueError(f"{stem}: plotting panel is {box.width:.3f} x {box.height:.3f} in")
    # Check complete text/legend bounds instead of silently clipping an export.
    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes]
    boxes += [legend.get_window_extent(renderer) for legend in fig.legends]
    for box in boxes:
        if box is not None and (box.x0 < -1 or box.y0 < -1 or
                                box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1):
            raise ValueError(f"{stem}: text or legend extends beyond the figure canvas")
    directory = Path(directory)
    for suffix in ("pdf", "png", "svg"):
        # Cropping trims outer whitespace; it does not rescale the 4.5-inch axes.
        fig.savefig(directory / f"{stem}.{suffix}", dpi=220, bbox_inches="tight", pad_inches=.12)
