"""Diagnostics for Improvement 1.  Place at scripts/cand_check.py

    .venv/bin/python scripts/cand_check.py --games ls20,ft09,s5i5,sb26,bp35 --steps 80

For each game it runs a random agent, then reports
  * click-candidate set size (objects+grid / objects only / grid only) on the first frame
  * the frame-change rate, and the rate after masking cells that change in >=90% of steps.
    If the masked rate collapses (e.g. 1.0 -> 0.1) the "always changes" games are driven by
    a HUD/timer, so frame-change is NOT a usable effect signal there (evidence for Improvement 2).
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
import arc_agi
from arc_agi import OperationMode

p = argparse.ArgumentParser()
p.add_argument("--games", default="ls20,ft09,s5i5,sb26,bp35")
p.add_argument("--steps", type=int, default=80)
args = p.parse_args()

spec = importlib.util.spec_from_file_location("user_agent_module", ROOT / "agent" / "my_agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
os.environ["ARC_MODE"] = "random"
mod.MyAgent.MAX_ACTIONS = args.steps
if not hasattr(mod, "click_candidates"):
    raise SystemExit("agent/my_agent.py is not the v2 file (no click_candidates)")

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
print(f"{'game':6} {'cand obj+grid':>13} {'obj only':>9} {'grid only':>10} "
      f"{'chg rate':>9} {'masked':>7} {'n always-chg cells':>19}  bbox(y0,y1,x0,x1)")
for gid in args.games.split(","):
    env = arc.make(gid, render_mode=None)
    ag = mod.MyAgent(card_id="diag", game_id=gid, agent_name=f"diag.{gid}", ROOT_URL="http://localhost",
                     record=False, arc_env=env, tags=["diag"])
    ag.main()
    obs = [np.asarray(f.frame[-1], dtype=np.uint8) for f in ag.frames if f.frame]
    o0 = obs[1] if len(obs) > 1 else obs[0]
    c_all = int(mod.click_candidates(o0).sum())
    c_obj = int(mod.click_candidates(o0, use_grid=False).sum())
    c_grid = int(mod.click_candidates(o0, max_comp=0).sum())
    diffs = np.stack([a != b for a, b in zip(obs, obs[1:])])        # (T,64,64)
    freq = diffs.mean(0)
    hud = freq >= 0.9
    rate = float(diffs.any((1, 2)).mean())
    masked = float((diffs & ~hud).any((1, 2)).mean())
    if hud.any():
        ys, xs = np.where(hud)
        bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
    else:
        bbox = None
    print(f"{gid:6} {c_all:13d} {c_obj:9d} {c_grid:10d} {rate:9.2f} {masked:7.2f} {int(hud.sum()):19d}  {bbox}")