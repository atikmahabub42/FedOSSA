#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_results.py -- tables and figures from the result CSVs.

    python make_results.py                      # reads results/part_*/ (seeds 0, 1, 2)
    python make_results.py --seeds 0            # seed 0 only

Writes into <root>/summary/:
    table_main.csv            test PPL, CIDEr-D, final round training loss per LLM/method
    training_loss.csv         mean training loss per round (plotted values)
    noise.csv, stragglers.csv perplexity versus rho / phi (plotted values)
    fig_training_loss.pdf/.eps
    fig_robustness.pdf/.eps   (a) rho, (b) phi: solid = equal weights, dashed = staleness

Rules: values are means over the selected seeds and the three datasets; a cell is
reported only when every (seed, dataset) of that cell exists. For the straggler figure,
each LLM uses only the (seed, dataset) pairs that every plotted method completed over the
whole phi grid, so all curves in a panel share the same runs. Nothing is interpolated.
"""
import argparse
import csv
import glob
import json
import logging
import os
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

MODELS = {"gpt2-l": "GPT-2 Large", "bloom-1b1": "BLOOM-1.1B", "qwen3-1.7b": "Qwen3-1.7B"}
DATASETS = ["dolly", "alpaca", "natinst"]
LABEL = {"centralized": "Centralized", "fedavg": "FedAvg / FedIT", "fedit": "FedIT",
         "fedptuning": "FedPTuning", "fedask": "FedASK", "fedossa": "FedOSSA",
         "fedossa_secagg": "FedOSSA+SecAgg"}
#            colour     marker  linestyle                    width  z  filled
STYLE = {"fedossa_secagg": ("#2a78d6", "o", "-",                    2.3, 6, True),
         "fedossa":        ("#008300", "o", (0, (1.2, 1.2)),        1.4, 7, False),
         "fedask":         ("#1baf7a", "^", (0, (5, 1.5, 1, 1.5)),  1.5, 4, True),
         "fedavg":         ("#eda100", "D", (0, (2.5, 1.2)),        1.5, 3, True),
         "fedit":          ("#6250d6", "P", (0, (3, 1, 1, 1)),      1.5, 3, True),
         "fedptuning":     ("#e87ba4", "v", ":",                    1.7, 2, True),
         "centralized":    ("#52514e", None, (0, (6, 2)),           1.5, 1, True)}
RHO = [0.0, 0.2, 0.4, 0.6, 0.8]
PHI = [0.25, 0.5, 0.75]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "font.weight": "bold", "axes.linewidth": 0.8,
    "axes.labelweight": "bold", "axes.titleweight": "bold", "mathtext.default": "bf",
    "axes.grid": True, "grid.color": "#e3e3e0", "grid.linewidth": 0.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "axes.edgecolor": "#3a3936", "xtick.color": "#1f1f1d", "ytick.color": "#1f1f1d",
    "pdf.fonttype": 42, "ps.fonttype": 42})


# ----------------------------------------------------------------- reading --------
def read_rows(root, mk, name):
    rows = []
    base = os.path.join(root, f"part_{mk}", name)
    for p in [base] + sorted(glob.glob(base[:-4] + "_recovered_*.csv")):
        if os.path.exists(p):
            rows += list(csv.DictReader(open(p, newline="")))
    return rows


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def cell_mean(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def methods_in(rows, allowed):
    present = {r["method"] for r in rows}
    return [m for m in LABEL if m in present and m in allowed]


# ----------------------------------------------------------------- tables ---------
def main_table(root, models, seeds):
    out = []
    for mk in models:
        rows = read_rows(root, mk, "sweep_results.csv")
        for m in methods_in(rows, LABEL):
            vals = {}
            for col in ("perplexity", "cider", "train_loss_final"):
                cell = {(r["seed"], r["dataset"]): fnum(r[col]) for r in rows if r["method"] == m}
                vals[col] = cell_mean([cell.get((str(s), d)) for s in seeds for d in DATASETS])
            out.append([MODELS[mk], LABEL[m]] +
                       ["--" if vals[c] is None else f"{vals[c]:.4f}"
                        for c in ("perplexity", "cider", "train_loss_final")])
    return out


def curves(root, models, seeds):
    data = {}
    for mk in models:
        p = os.path.join(root, f"part_{mk}", "sweep_curves.json")
        if not os.path.exists(p):
            continue
        raw = json.load(open(p))
        per = defaultdict(dict)
        for key, series in raw.items():
            parts = key.split("|")                       # s{seed}|{dataset}|{model}|{method}
            if len(parts) >= 4 and parts[0][1:] in {str(s) for s in seeds}:
                per[parts[3]][(parts[0], parts[1])] = series
        data[mk] = {}
        for m, cells in per.items():
            need = [(f"s{s}", d) for s in seeds for d in DATASETS]
            if not all(c in cells for c in need):
                continue
            n = min(len(cells[c]) for c in need)
            data[mk][m] = [float(np.mean([cells[c][i]["train_loss"] for c in need]))
                           for i in range(n)]
    return data


def noise(root, models, seeds):
    data = {}
    for mk in models:
        cell = {}
        for r in read_rows(root, mk, "dp_sweep.csv"):
            cell[(r["method"], r["seed"], r["dataset"], float(r["dp_sigma"]))] = fnum(r["perplexity"])
        ms = sorted({k[0] for k in cell}, key=lambda m: list(LABEL).index(m) if m in LABEL else 99)
        data[mk] = {m: [cell_mean([cell.get((m, str(s), d, rho)) for s in seeds for d in DATASETS])
                        for rho in RHO] for m in ms if m in LABEL}
    return data


def stragglers(root, models, seeds):
    data, used = {}, {}
    grid = [(0.0, "sync")] + [(f, md) for f in PHI for md in ("async_naive", "async_weighted")]
    for mk in models:
        cell = {}
        for r in read_rows(root, mk, "async.csv"):
            cell[(r["method"], r["seed"], r["dataset"], float(r["straggler_frac"]), r["mode"])] = \
                fnum(r["perplexity"])
        ms = [m for m in LABEL if any(k[0] == m for k in cell)]
        pairs = [(str(s), d) for s in seeds for d in DATASETS
                 if all(cell.get((m, str(s), d, f, md)) is not None for m in ms for f, md in grid)]
        used[mk] = pairs
        data[mk] = {m: {(f, md): cell_mean([cell[(m, s, d, f, md)] for s, d in pairs]) if pairs else None
                        for f, md in grid} for m in ms}
    return data, used


# ----------------------------------------------------------------- figures --------
def _line(ax, m, x, y, ls=None, ms=3.0):
    c, mk, dls, lw, z, filled = STYLE[m]
    pts = [(a, b) for a, b in zip(x, y) if b is not None]
    if not pts:
        return
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=c, ls=ls or dls, lw=lw, zorder=z, marker=mk, markersize=ms,
            markerfacecolor=c if filled else "white", markeredgecolor=c, markeredgewidth=0.8)


def _handles(methods):
    h = []
    for m in methods:
        c, mk, ls, lw, z, filled = STYLE[m]
        h.append(Line2D([0], [0], color=c, ls=ls, lw=lw, marker=mk, markersize=4.2,
                        markerfacecolor=c if filled else "white", markeredgecolor=c,
                        label=LABEL[m]))
    return h


def fig_training(data, out):
    models = [mk for mk in MODELS if data.get(mk)]
    if not models:
        return
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.05))
    cells = [axs[0][0], axs[0][1], axs[1][0]]
    shown = []
    for ax, mk in zip(cells, models):
        for m in LABEL:
            if m in data[mk]:
                y = data[mk][m]
                _line(ax, m, list(range(1, len(y) + 1)), y)
                shown.append(m)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(MODELS[mk], fontsize=8.5, pad=2)
        ax.set_xlabel("Communication round", fontsize=8)
        ax.set_ylabel("Training loss", fontsize=8)
    axs[1][1].axis("off")
    axs[1][1].legend(handles=_handles([m for m in LABEL if m in shown]), loc="center",
                     frameon=False, fontsize=7.8, handlelength=2.6)
    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.5)
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(out, f"fig_training_loss.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def _broken(fig, spec):
    gs = spec.subgridspec(2, 1, height_ratios=[1, 2.1], hspace=0.08)
    t = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1], sharex=t)
    t.spines["bottom"].set_visible(False); b.spines["top"].set_visible(False)
    t.tick_params(bottom=False, labelbottom=False)
    for ax, y, sg in ((t, 0.0, 1), (b, 1.0, -1)):
        for x0 in (0.0, 1.0):
            ax.plot([x0 - 0.025, x0 + 0.025], [y - 0.06 * sg, y + 0.06 * sg],
                    transform=ax.transAxes, color="#3a3936", clip_on=False, lw=0.7)
    return t, b


def _pad(ax, ys, nb):
    ys = [v for v in ys if v is not None]
    if not ys:
        ax.set_visible(False)
        return
    lo, hi = min(ys), max(ys)
    d = (hi - lo) or abs(hi) * 0.01
    ax.set_ylim(lo - 0.14 * d, hi + 0.14 * d)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=nb))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))


def fig_robustness(nz, st, out):
    models = [mk for mk in MODELS if nz.get(mk) or st.get(mk)]
    if not models:
        return
    fig = plt.figure(figsize=(3.5, 1.65 * len(models) + 0.25))
    outer = fig.add_gridspec(len(models), 2, hspace=0.42, wspace=0.42, left=0.15,
                             right=0.985, top=1 - 0.6 / (1.65 * len(models) + 0.25), bottom=0.07)
    methods = set()
    for i, mk in enumerate(models):
        for j in range(2):
            t, b = _broken(fig, outer[i, j])
            top_v, bot_v = [], []
            src = nz.get(mk, {}) if j == 0 else st.get(mk, {})
            for m in [m for m in LABEL if m in src and m != "centralized"]:
                methods.add(m)
                ax = t if m == "fedptuning" else b
                if j == 0:
                    _line(ax, m, RHO, src[m]); ys = src[m]
                else:
                    x = [0.0] + PHI
                    ys = []
                    for mode, ls in (("async_naive", "-"), ("async_weighted", (0, (3.2, 1.6)))):
                        y = [src[m][(0.0, "sync")]] + [src[m][(f, mode)] for f in PHI]
                        _line(ax, m, x, y, ls=ls); ys += y
                (top_v if m == "fedptuning" else bot_v).extend(ys)
            _pad(t, top_v, 2); _pad(b, bot_v, 3)
            b.set_xticks([0, 0.2, 0.4, 0.6, 0.8] if j == 0 else [0, 0.25, 0.5, 0.75])
            if j == 0:
                b.set_xticklabels(["0", ".2", ".4", ".6", ".8"])
            t.set_title(MODELS[mk], fontsize=8, pad=2)
            if i == len(models) - 1:
                b.set_xlabel(r"(a) noise level $\rho$" if j == 0 else r"(b) straggler fraction $\phi$",
                             fontsize=8)
            if j == 0:
                b.set_ylabel("Test PPL", fontsize=7.6); t.set_ylabel("FedPT.", fontsize=6.6)
    h = _handles([m for m in LABEL if m in methods])
    h += [Line2D([0], [0], color="#3a3936", ls="-", lw=1.4, label="(b) equal weight"),
          Line2D([0], [0], color="#3a3936", ls=(0, (3.2, 1.6)), lw=1.1, label="(b) staleness weight")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=6.9,
               bbox_to_anchor=(0.54, 1.0), handlelength=2.2, columnspacing=0.8)
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(out, f"fig_robustness.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="tables and figures from the result CSVs")
    ap.add_argument("--root", default="results")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    models = [mk for mk in MODELS if os.path.isdir(os.path.join(a.root, f"part_{mk}"))]
    if not models:
        print(f"no part_<model> folders under {a.root} -- run run_sweep.py first")
        return 1
    out = os.path.join(a.root, "summary")
    os.makedirs(out, exist_ok=True)

    tab = main_table(a.root, models, a.seeds)
    write_csv(os.path.join(out, "table_main.csv"),
              ["llm", "method", "test_ppl", "cider_d", "final_train_loss"], tab)
    print(f"{'LLM':<13}{'Method':<17}{'PPL':>10}{'CIDEr-D':>10}{'Loss':>10}")
    for r in tab:
        print(f"{r[0]:<13}{r[1]:<17}{r[2]:>10}{r[3]:>10}{r[4]:>10}")

    tr = curves(a.root, models, a.seeds)
    write_csv(os.path.join(out, "training_loss.csv"), ["llm", "method", "round", "train_loss"],
              [[mk, m, i + 1, round(v, 5)] for mk in tr for m in tr[mk] for i, v in enumerate(tr[mk][m])])
    nz = noise(a.root, models, a.seeds)
    write_csv(os.path.join(out, "noise.csv"), ["llm", "method", "rho", "test_ppl"],
              [[mk, m, rho, "" if v is None else round(v, 4)] for mk in nz for m in nz[mk]
               for rho, v in zip(RHO, nz[mk][m])])
    st, used = stragglers(a.root, models, a.seeds)
    write_csv(os.path.join(out, "stragglers.csv"),
              ["llm", "method", "phi", "mode", "test_ppl", "runs_used"],
              [[mk, m, f, md, "" if v is None else round(v, 4),
                " ".join(f"s{s}/{d}" for s, d in used[mk])]
               for mk in st for m in st[mk] for (f, md), v in st[mk][m].items()])
    fig_training(tr, out)
    fig_robustness(nz, st, out)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
