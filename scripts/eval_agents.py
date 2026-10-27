"""Evaluate agent variants on a small, disclosed subset of ARC-AGI-3 games.

Place at scripts/eval_agents.py. It drives agent/my_agent.py exactly the way
scripts/play_local.py does, but loops over variants x seeds x games and writes
machine-readable results for the report.

Examples
    # show which 5 dev games will be used, then exit
    .venv/bin/python scripts/eval_agents.py --dry-run

    # random vs DQN, 3 seeds, 500 actions per game, on the 5 dev games
    .venv/bin/python scripts/eval_agents.py --variant random --variant dqn \
        --seeds 0,1,2 --max-steps 500 --out results/baseline

    # an ablation: DQN with a faster epsilon decay (any ARC_* env var works)
    .venv/bin/python scripts/eval_agents.py --variant dqn_fast:ARC_MODE=dqn,ARC_EPS_DECAY=300 \
        --seeds 0,1,2 --out results/baseline --resume

    # held-out games: run ONCE per final variant, after development is finished
    .venv/bin/python scripts/eval_agents.py --split heldout --variant dqn --out results/heldout

A bare variant name ("random", "dqn") means ARC_MODE=<name>. Otherwise use
NAME:KEY=VAL,KEY=VAL. Each (variant, seed) gets a fresh Arcade, so its
scorecard covers only that run. Local scores are NOT Kaggle scores.

Outputs in --out
    manifest.json   games, dev/held-out split, seeds, budgets, versions, hardware
    runs.csv        one row per (variant, seed, game), failed runs included
    aggregates.csv  one row per (variant, seed): official local score + totals
    curves.csv      levels / unique states / game-overs vs. actions taken
    summary.csv     per-variant mean/std across seeds
    scorecards/     raw official scorecard per (variant, seed)

Proxy metrics (unique states, frame-change rate, game-overs) are diagnostics,
not the ARC-AGI-3 score. Frame index is used as the action index (they match
up to the initial frame).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
AGENT_PATH = ROOT / "agent" / "my_agent.py"
ALWAYS_DEV = ["ls20", "ft09"]       # already used during development, so they are dev games

RUN_FIELDS = ["variant", "seed", "game", "state", "levels_final", "levels_max", "actions",
              "game_overs", "unique_states", "frame_change_rate", "level_up_actions",
              "seconds", "steps_per_sec", "simple_action_counts", "distinct_click_pixels", "error"]
AGG_FIELDS = ["variant", "seed", "score", "total_levels", "games_with_level", "total_actions",
              "total_game_overs", "total_unique_states", "seconds", "n_games", "n_errors"]
CURVE_FIELDS = ["variant", "seed", "game", "step", "levels", "unique_states", "game_overs"]


# ---------------------------------------------------------------- game selection
def select_games(all_ids, n=5, split_seed=0):
    """Deterministic dev / held-out split. Returns (dev, heldout), disjoint."""
    pool = sorted(set(all_ids))
    dev = [g for g in ALWAYS_DEV if g in pool][:n]
    rest = [g for g in pool if g not in dev]
    dev += random.Random(split_seed).sample(rest, min(n - len(dev), len(rest)))
    remaining = [g for g in pool if g not in dev]
    held = random.Random(split_seed + 1).sample(remaining, min(n, len(remaining)))
    return dev, held


# ---------------------------------------------------------------- metrics
def _is_game_over(f) -> bool:
    return str(getattr(f, "state", "")).endswith("GAME_OVER")


def analyse_frames(frames, checkpoints):
    hashes, level_ups, over_flags = [], [], []
    run_max = 0
    for i, f in enumerate(frames):
        try:
            hashes.append(hash(np.asarray(f.frame[-1]).tobytes()))
        except Exception:
            hashes.append(None)
        lv = int(getattr(f, "levels_completed", 0) or 0)
        if lv > run_max:
            level_ups.append(i)
            run_max = lv
        over_flags.append(_is_game_over(f))
    n = len(frames)
    changes = sum(1 for a, b in zip(hashes, hashes[1:]) if a != b)
    curves = []
    for cp in checkpoints:
        if n == 0:
            break
        idx = min(cp, n - 1)
        curves.append({
            "step": cp if cp < n else n - 1,
            "levels": sum(1 for u in level_ups if u <= idx),
            "unique_states": len({h for h in hashes[:idx + 1] if h is not None}),
            "game_overs": sum(over_flags[:idx + 1]),
        })
    return {
        "levels_max": run_max,
        "game_overs": sum(over_flags),
        "unique_states": len({h for h in hashes if h is not None}),
        "frame_change_rate": round(changes / max(1, n - 1), 4),
        "level_up_actions": ";".join(map(str, level_ups)),
        "curves": curves,
    }


def agent_stats(agent) -> dict:
    s = getattr(agent, "stats", None)
    if not isinstance(s, dict) or "acts" not in s:
        return {"simple_action_counts": "", "distinct_click_pixels": ""}
    acts = np.asarray(s["acts"], dtype=int)
    return {
        "simple_action_counts": "/".join(str(int(x)) for x in acts[:5]),
        "distinct_click_pixels": int((acts[5:] > 0).sum()),
    }


def scorecard_dict(sc) -> dict:
    for attr in ("model_dump", "dict"):
        fn = getattr(sc, attr, None)
        if callable(fn):
            try:
                return json.loads(json.dumps(fn(), default=str))
            except Exception:
                pass
    return {"repr": repr(sc)}


# ---------------------------------------------------------------- summary
def _ms(x):
    if not x:
        return float("nan"), float("nan")
    return statistics.mean(x), (statistics.stdev(x) if len(x) > 1 else 0.0)


def summarise(aggs):
    rows = []
    for v in sorted({a["variant"] for a in aggs}):
        rs = [a for a in aggs if a["variant"] == v]
        scores = [float(a["score"]) for a in rs if str(a["score"]) not in ("", "None")]
        levels = [float(a["total_levels"]) for a in rs]
        sm, ss = _ms(scores)
        lm, ls = _ms(levels)
        rows.append({
            "variant": v, "n_seeds": len(rs),
            "score_mean": round(sm, 4), "score_std": round(ss, 4),
            "levels_mean": round(lm, 3), "levels_std": round(ls, 3),
            "seeds_with_a_level": sum(1 for x in levels if x > 0),
            "unique_states_mean": round(_ms([float(a["total_unique_states"]) for a in rs])[0], 1),
            "game_overs_mean": round(_ms([float(a["total_game_overs"]) for a in rs])[0], 2),
            "seconds_mean": round(_ms([float(a["seconds"]) for a in rs])[0], 1),
            "errors": sum(int(a["n_errors"]) for a in rs),
        })
    return rows


# ---------------------------------------------------------------- io helpers
def append_csv(path: Path, fields, rows):
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def parse_variant(spec: str):
    if ":" not in spec:
        return spec, {"ARC_MODE": spec}
    name, rest = spec.split(":", 1)
    env = {}
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        env[k.strip()] = v.strip()
    return name, env


@contextlib.contextmanager
def env_overrides(env: dict):
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def versions() -> dict:
    from importlib import metadata
    out = {"python": platform.python_version()}
    for pkg in ("arc-agi", "arcengine", "numpy", "torch"):
        try:
            out[pkg] = metadata.version(pkg)
        except Exception:
            out[pkg] = "n/a"
    try:
        import torch
        out["cuda"] = bool(torch.cuda.is_available())
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    out["cpu"] = platform.processor() or platform.machine()
    out["cpu_count"] = os.cpu_count()
    return out


def load_agent_class():
    spec = importlib.util.spec_from_file_location("user_agent_module", AGENT_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {AGENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MyAgent"):
        raise SystemExit(f"{AGENT_PATH} must define a class named MyAgent")
    return module.MyAgent


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", action="append", help="repeatable; 'dqn' or 'name:KEY=VAL,...'")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--max-steps", type=int, default=500, help="action cap per game, identical for every variant")
    p.add_argument("--n-games", type=int, default=5)
    p.add_argument("--split", choices=["dev", "heldout"], default="dev")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--games", default=None, help="explicit comma-separated list; overrides --split")
    p.add_argument("--checkpoints", default="100,250,500,1000,2000")
    p.add_argument("--out", default=None)
    p.add_argument("--resume", action="store_true", help="skip (variant, seed) pairs already in aggregates.csv")
    p.add_argument("--dry-run", action="store_true", help="print the game split and exit")
    args = p.parse_args()

    if not VENDOR.exists():
        raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(VENDOR))
    import arc_agi
    from arc_agi import OperationMode

    probe = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_ids = [e.game_id.split("-")[0] for e in probe.get_environments()]
    dev, held = select_games(all_ids, args.n_games, args.split_seed)
    if args.games:
        games = [g.strip().split("-")[0] for g in args.games.split(",")]
        unknown = set(games) - set(all_ids)
        if unknown:
            raise SystemExit(f"Unknown game ids: {sorted(unknown)}")
    else:
        games = dev if args.split == "dev" else held
    print(f"split={args.split}\n  games      : {games}\n  dev        : {dev}\n  held-out   : {held}")
    if args.dry_run:
        return

    variants = [parse_variant(v) for v in (args.variant or ["random", "dqn"])]
    seeds = [int(s) for s in args.seeds.split(",")]
    checkpoints = sorted({int(c) for c in args.checkpoints.split(",") if int(c) <= args.max_steps} | {args.max_steps})
    out = Path(args.out or f"results/eval_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}")
    (out / "scorecards").mkdir(parents=True, exist_ok=True)
    manifest = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "split": args.split,
        "games": games, "dev_games": dev, "heldout_games": held, "split_seed": args.split_seed,
        "variants": {n: e for n, e in variants}, "seeds": seeds, "max_steps": args.max_steps,
        "checkpoints": checkpoints, "versions": versions(),
        "note": "local scores from the unmodified arc_agi scorecard; not Kaggle scores",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    done = {(a["variant"], int(a["seed"])) for a in read_csv(out / "aggregates.csv")} if args.resume else set()
    AgentCls = load_agent_class()
    AgentCls.MAX_ACTIONS = args.max_steps          # same action budget for every variant

    for vname, venv in variants:
        for seed in seeds:
            if (vname, seed) in done:
                print(f"skip {vname} seed={seed} (already done)")
                continue
            print(f"\n######## variant={vname} seed={seed} env={venv}")
            arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)   # fresh scorecard per run
            run_rows, curve_rows = [], []
            t_run = time.time()
            with env_overrides(venv):
                for gi, gid in enumerate(games):
                    os.environ["ARC_SEED"] = str(seed * 1000 + gi)        # distinct, reproducible per game
                    row = {"variant": vname, "seed": seed, "game": gid, "error": ""}
                    t0 = time.time()
                    try:
                        env = arc.make(gid, render_mode=None)
                        if env is None:
                            raise RuntimeError("arc.make returned None")
                        agent = AgentCls(card_id=f"eval-{vname}-{seed}", game_id=gid,
                                         agent_name=f"{vname}.s{seed}.{gid}", ROOT_URL="http://localhost",
                                         record=False, arc_env=env, tags=["eval", vname])
                        agent.main()
                        final = agent.frames[-1]
                        m = analyse_frames(agent.frames, checkpoints)
                        secs = time.time() - t0
                        row.update(state=str(final.state), levels_final=int(final.levels_completed),
                                   actions=int(agent.action_counter), seconds=round(secs, 1),
                                   steps_per_sec=round(agent.action_counter / max(secs, 1e-9), 2),
                                   **{k: v for k, v in m.items() if k != "curves"}, **agent_stats(agent))
                        curve_rows += [{"variant": vname, "seed": seed, "game": gid, **c} for c in m["curves"]]
                    except Exception as exc:                  # keep failed runs in the table
                        traceback.print_exc()
                        row.update(state="ERROR", levels_final=0, levels_max=0, actions=0,
                                   seconds=round(time.time() - t0, 1), error=repr(exc)[:200])
                    run_rows.append(row)
                    print(f"  -> {gid}: state={row['state']} levels={row.get('levels_final')} "
                          f"actions={row.get('actions')}")

            sc = arc.get_scorecard()
            score = sc.score if hasattr(sc, "score") else sc
            (out / "scorecards" / f"{vname}_seed{seed}.json").write_text(
                json.dumps(scorecard_dict(sc), indent=2, default=str))
            agg = {
                "variant": vname, "seed": seed, "score": score,
                "total_levels": sum(int(r.get("levels_final", 0)) for r in run_rows),
                "games_with_level": sum(1 for r in run_rows if int(r.get("levels_final", 0)) > 0),
                "total_actions": sum(int(r.get("actions", 0)) for r in run_rows),
                "total_game_overs": sum(int(r.get("game_overs", 0) or 0) for r in run_rows),
                "total_unique_states": sum(int(r.get("unique_states", 0) or 0) for r in run_rows),
                "seconds": round(time.time() - t_run, 1), "n_games": len(run_rows),
                "n_errors": sum(1 for r in run_rows if r["state"] == "ERROR"),
            }
            # written together at the end of a (variant, seed) so a crash never leaves half a run
            append_csv(out / "runs.csv", RUN_FIELDS, run_rows)
            append_csv(out / "curves.csv", CURVE_FIELDS, curve_rows)
            append_csv(out / "aggregates.csv", AGG_FIELDS, [agg])
            print(f"  == {vname} seed={seed}: local score={score} levels={agg['total_levels']} "
                  f"time={agg['seconds']}s")

    summary = summarise(read_csv(out / "aggregates.csv"))
    if summary:
        with (out / "summary.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        print("\n================ SUMMARY (local scores, mean +/- std over seeds) ================")
        for r in summary:
            print(f"{r['variant']:14} seeds={r['n_seeds']}  score={r['score_mean']}+/-{r['score_std']}  "
                  f"levels={r['levels_mean']}+/-{r['levels_std']}  seeds_with_level={r['seeds_with_a_level']}  "
                  f"unique_states={r['unique_states_mean']}  game_overs={r['game_overs_mean']}  "
                  f"sec={r['seconds_mean']}  errors={r['errors']}")
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()