"""Plot adaptation curves from eval_agents.py output.  Place at scripts/plot_results.py

    .venv/bin/pip install pandas matplotlib
    .venv/bin/python scripts/plot_results.py results/baseline
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

out = Path(sys.argv[1])
df = pd.read_csv(out / "curves.csv")
# sum over games -> one value per (variant, seed, step)
per_seed = df.groupby(["variant", "seed", "step"], as_index=False)[
    ["levels", "unique_states", "game_overs"]].sum()

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, col, title in zip(axes, ["levels", "unique_states", "game_overs"],
                          ["Levels completed (sum over games)", "Unique frames seen", "Game-overs"]):
    for v, g in per_seed.groupby("variant"):
        piv = g.pivot(index="step", columns="seed", values=col)
        m, s = piv.mean(1), piv.std(1).fillna(0)
        ax.plot(m.index, m.values, marker="o", label=v)
        ax.fill_between(m.index, m - s, m + s, alpha=0.2)
        for sd in piv.columns:                       # individual seeds, faint
            ax.plot(piv.index, piv[sd], color="gray", alpha=0.25, lw=0.8)
    ax.set_xlabel("actions taken"); ax.set_title(title)
axes[0].legend()
plt.tight_layout()
plt.savefig(out / "curves.pdf")
plt.savefig(out / "curves.png", dpi=150)
print("wrote", out / "curves.pdf")