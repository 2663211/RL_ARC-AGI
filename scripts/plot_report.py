"""All report figures from eval_agents.py outputs.  Place at scripts/plot_report.py

    .venv/bin/pip install pandas matplotlib
    .venv/bin/python scripts/plot_report.py results/baseline [results/imp2a results/imp2b ...] --figs figs

Several result dirs are concatenated (use this if you ran variants in parallel).
Writes PDF+PNG figures to --figs:
  fig_curves      adaptation curves vs actions (levels, unique frames, game-overs), main ladder
  fig_games       per-game bars + individual seeds (unique frames, frame-change), click games
  fig_ablation    Improvement-2 ablation incl. random+memory controls, per game
  fig_score       per-seed aggregate local score and levels (shows how noisy they are)
  fig_levels      when each level was solved (action index), one dot per solved run
  fig_sb26_type   share of ACTION5 on sb26 (click-bias diagnostic)
Variants that are missing from the data are skipped silently.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--figs", default="figs")
args = ap.parse_args()
out = Path(args.figs)
out.mkdir(parents=True, exist_ok=True)


def load(name):
    fr = [pd.read_csv(Path(d) / name) for d in args.dirs if (Path(d) / name).exists()]
    return pd.concat(fr, ignore_index=True).drop_duplicates() if fr else pd.DataFrame()


runs, aggs, curves = load("runs.csv"), load("aggregates.csv"), load("curves.csv")

LADDER = ["random", "dqn", "dqn_obj", "dqn_obj_mem"]
ABL = ["random_obj", "random_obj_mem", "dqn_obj", "dqn_mem", "dqn_obj_mem"]
ALL = ["random", "dqn", "dqn_nonovelty", "dqn_obj", "dqn_obj_nogrid", "dqn_gridonly",
       "random_obj", "random_obj_mem", "dqn_mem", "dqn_obj_mem"]
CLICK = ["ft09", "sb26", "s5i5", "bp35"]
COL = {v: plt.cm.tab10(i) for i, v in enumerate(ALL)}
have = set(runs.variant) if len(runs) else set()


def keep(vs):
    return [v for v in vs if v in have]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=140)
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def bars(ax, df, games, variants, col, title):
    w = 0.8 / max(1, len(variants))
    for gi, g in enumerate(games):
        for vi, v in enumerate(variants):
            y = df[(df.game == g) & (df.variant == v)][col].astype(float).values
            if not len(y):
                continue
            x = gi + (vi - (len(variants) - 1) / 2) * w
            ax.bar(x, y.mean(), width=w * 0.92, color=COL[v], label=v if gi == 0 else None)
            ax.scatter([x] * len(y), y, s=6, color="k", zorder=3)
    ax.set_xticks(range(len(games)))
    ax.set_xticklabels(games)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


# 1. adaptation curves (sum over games, mean +- std over seeds, grey = individual seeds)
if len(curves):
    vs = [v for v in LADDER if v in set(curves.variant)]
    per = curves.groupby(["variant", "seed", "step"], as_index=False)[
        ["levels", "unique_states", "game_overs"]].sum()
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, col, t in zip(axes, ["levels", "unique_states", "game_overs"],
                          ["Levels completed (sum over games)", "Unique frames", "Game-overs"]):
        for v in vs:
            piv = per[per.variant == v].pivot(index="step", columns="seed", values=col)
            m, s = piv.mean(axis=1), piv.std(axis=1).fillna(0)
            ax.plot(m.index, m.values, marker="o", ms=3, color=COL[v], label=v)
            ax.fill_between(m.index, m - s, m + s, color=COL[v], alpha=0.15)
            for sd in piv.columns:
                ax.plot(piv.index, piv[sd], color=COL[v], alpha=0.25, lw=0.7)
        ax.set_xlabel("actions taken", fontsize=8)
        ax.set_title(t, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, frameon=False)
    save(fig, "fig_curves")

# 2. per-game bars, click games
if len(runs):
    vs = keep(["random", "dqn", "dqn_obj", "dqn_obj_mem"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
    bars(axes[0], runs, CLICK, vs, "unique_states", "Unique frames (500 actions)")
    bars(axes[1], runs, CLICK, vs, "frame_change_rate", "Frame-change rate")
    axes[0].legend(fontsize=7, frameon=False)
    save(fig, "fig_games")

    # 3. Improvement-2 ablation (+ random controls)
    vs = keep(ABL)
    if len(vs) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
        g2 = ["ft09", "sb26"]
        bars(axes[0], runs, g2, vs, "frame_change_rate", "Frame-change rate")
        bars(axes[1], runs, g2, vs, "unique_states", "Unique frames")
        axes[0].legend(fontsize=7, frameon=False)
        save(fig, "fig_ablation")

    # 6. ACTION5 share on sb26
    s = runs[runs.game == "sb26"].copy()
    s["a5"] = s.simple_action_counts.astype(str).str.split("/").str[4].astype(float) / s.actions
    vs = keep(["random", "dqn", "dqn_nonovelty", "dqn_obj", "dqn_obj_mem", "dqn_mem"])
    fig, ax = plt.subplots(figsize=(5, 2.8))
    for i, v in enumerate(vs):
        y = s[s.variant == v].a5.values
        ax.bar(i, y.mean(), color=COL[v])
        ax.scatter([i] * len(y), y, s=8, color="k", zorder=3)
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels(vs, rotation=30, ha="right", fontsize=7)
    ax.set_title("sb26: share of ACTION5 (rest = clicks)", fontsize=9)
    save(fig, "fig_sb26_type")

    # 5. when levels were solved
    sol = runs[runs.levels_final.astype(float) > 0].copy()
    if len(sol):
        vs = keep(ALL)
        fig, ax = plt.subplots(figsize=(6.5, 2.8))
        for i, v in enumerate(vs):
            sub = sol[sol.variant == v]
            for _, r in sub.iterrows():
                first = float(str(r.level_up_actions).split(";")[0])
                ax.scatter(first, i, color=COL[v], s=30)
                ax.annotate(f"{r.game}/s{int(r.seed)}", (first, i), fontsize=6, xytext=(3, 3),
                            textcoords="offset points")
        ax.set_yticks(range(len(vs)))
        ax.set_yticklabels(vs, fontsize=7)
        ax.set_xlim(0, 520)
        ax.set_xlabel("action index of first level-up (smaller = more efficient)", fontsize=8)
        ax.set_title("Solved levels (variants with no dot solved none)", fontsize=9)
        save(fig, "fig_levels")

# 4. per-seed aggregate score and levels
if len(aggs):
    vs = [v for v in ALL if v in set(aggs.variant)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
    for ax, col, t in zip(axes, ["score", "total_levels"],
                          ["Local aggregate score (per seed)", "Levels solved (per seed)"]):
        for i, v in enumerate(vs):
            y = aggs[aggs.variant == v][col].astype(float).values
            ax.bar(i, y.mean(), color=COL[v], alpha=0.8)
            ax.scatter([i] * len(y), y, s=10, color="k", zorder=3)
        ax.set_xticks(range(len(vs)))
        ax.set_xticklabels(vs, rotation=35, ha="right", fontsize=7)
        ax.set_title(t, fontsize=9)
    save(fig, "fig_score")