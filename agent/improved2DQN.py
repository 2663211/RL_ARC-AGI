"""Baseline 1: vanilla DQN for ARC-AGI-3 (drop-in replacement for agent/my_agent.py).

Design (document all of this in the report):
  State        : last frame layer, 64x64 ints in [0,15], one-hot -> 16 channels.
  Actions      : 4101 = ACTION1-5 (simple) + ACTION6 clicked at each of 64*64 pixels.
                 RESET is handled by the wrapper, never chosen by the network.
  Network      : 3x conv(3x3) trunk. Per-pixel 1x1 conv head -> Q(click at x,y);
                 global-mean-pooled features -> linear head -> Q(ACTION1-5).
  Reward (training only, never reported as the score):
                 +1 level completed, -0.1 on GAME_OVER, optional novelty bonus.
  Learning     : online, 1 gradient step per env step, uniform replay,
                 target network, epsilon-greedy, Huber loss, masked max over
                 the *next* state's available actions.
  Persistence  : weights/replay persist across levels and resets WITHIN one game.
                 A fresh agent (random init) is built for every game and every run.
                 Nothing is shared between games or loaded from disk.

Config via env vars so eval_agents.py can run ablations without editing code:
  ARC_MODE (dqn|random) ARC_EPS_DECAY ARC_LR ARC_GAMMA ARC_BATCH ARC_TARGET_EVERY
  ARC_WARMUP ARC_NOVELTY (0|1) ARC_SEED
  Improvement 1 (structured click space):
  ARC_ACTION_SPACE (pixel|object)  pixel = baseline, all 4096 click targets
                                   object = click only at candidate cells
  ARC_MAXCOMP  max connected-component candidates per frame (0 = grid only)
  ARC_GRID     1 = also add a coarse stride-8 grid of fallback click cells
  Improvement 2 (no-op memory):
  ARC_MEMORY   1 = remember, per exact frame, which actions left the frame unchanged;
               never repeat them there, and explore untried actions first
"""
from __future__ import annotations

import os
import random
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

N_SIMPLE, GRID = 5, 64
N_CLICK = GRID * GRID
N_ACTIONS = N_SIMPLE + N_CLICK
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cfg(name: str, default, cast=float):
    return cast(os.environ.get(name, default))


def click_candidates(obs: np.ndarray, stride: int = 8, max_comp: int = 128,
                     use_grid: bool = True) -> np.ndarray:
    """Bool mask over the 4096 click pixels (index = y*64 + x).

    Uses only the observed frame (no hidden state, no game-specific rules):
      * background = most frequent colour;
      * every 4-connected same-colour non-background component contributes ONE
        candidate: the component cell nearest its centroid;
      * optionally a coarse grid of fallback cells (stride 8 -> 64 cells) so
        clickable regions that are background-coloured stay reachable.
    """
    h, w = obs.shape
    mask = np.zeros(h * w, bool)
    if max_comp > 0:
        lst = obs.tolist()
        bg = int(np.bincount(obs.ravel(), minlength=16).argmax())
        seen = [[False] * w for _ in range(h)]
        pts = []
        for y0 in range(h):
            row = lst[y0]
            for x0 in range(w):
                c = row[x0]
                if c == bg or seen[y0][x0]:
                    continue
                seen[y0][x0] = True
                stack, cells = [(y0, x0)], []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and lst[ny][nx] == c:
                            seen[ny][nx] = True
                            stack.append((ny, nx))
                cy = sum(p[0] for p in cells) / len(cells)
                cx = sum(p[1] for p in cells) / len(cells)
                pts.append(min(cells, key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2))
        if len(pts) > max_comp:
            pts = random.sample(pts, max_comp)
        for y, x in pts:
            mask[y * w + x] = True
    if use_grid:
        for y in range(stride // 2, h, stride):
            for x in range(stride // 2, w, stride):
                mask[y * w + x] = True
    if not mask.any():
        mask[:] = True
    return mask


# --------------------------------------------------------------------------- replay
class ReplayBuffer:
    """Ring buffer in uint8 (a 64x64 obs is 4 KB, not 32 KB like int64)."""

    def __init__(self, cap: int = 10_000):
        self.cap, self.n, self.i = cap, 0, 0
        self.s = np.zeros((cap, GRID, GRID), np.uint8)
        self.s2 = np.zeros((cap, GRID, GRID), np.uint8)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.m2 = np.zeros((cap, N_ACTIONS), bool)   # valid actions in the next state

    def push(self, s, a, r, s2, d, m2):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i], self.m2[i] = s, a, r, s2, d, m2
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, k: int):
        idx = np.random.randint(0, self.n, size=k)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx], self.m2[idx])


# --------------------------------------------------------------------------- network
class QNet(nn.Module):
    def __init__(self, ch: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(16, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
        )
        self.click = nn.Conv2d(ch, 1, 1)             # Q(ACTION6 at pixel)
        self.simple = nn.Linear(ch, N_SIMPLE)        # Q(ACTION1..5)

    def forward(self, x):                            # x: (B,64,64) long
        x = F.one_hot(x, 16).permute(0, 3, 1, 2).float()
        h = self.conv(x)
        return torch.cat([self.simple(h.mean((2, 3))), self.click(h).flatten(1)], 1)


# --------------------------------------------------------------------------- agent
class MyAgent(Agent):
    MAX_ACTIONS = 500

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.MODE = os.environ.get("ARC_MODE", "dqn")
        self.GAMMA = _cfg("ARC_GAMMA", 0.99)
        self.LR = _cfg("ARC_LR", 1e-4)
        self.BATCH = _cfg("ARC_BATCH", 32, int)
        self.EPS_START, self.EPS_END = 1.0, 0.05
        self.EPS_DECAY = _cfg("ARC_EPS_DECAY", 250, int)     # linear, in env steps
        self.TARGET_EVERY = _cfg("ARC_TARGET_EVERY", 100, int)
        self.WARMUP = _cfg("ARC_WARMUP", 64, int)
        self.NOVELTY = _cfg("ARC_NOVELTY", 1, int)
        self.ASPACE = os.environ.get("ARC_ACTION_SPACE", "pixel")
        self.MAXCOMP = _cfg("ARC_MAXCOMP", 128, int)
        self.GRIDFB = _cfg("ARC_GRID", 1, int)
        self.MEM = _cfg("ARC_MEMORY", 0, int)
        self.noop = defaultdict(set)      # frame-hash -> actions that did not change the frame
        self.tried = defaultdict(set)     # frame-hash -> actions already taken in that frame

        seed = _cfg("ARC_SEED", int(time.time() * 1e6) % 2**31, int)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        self.q = QNet().to(DEV)
        self.tq = QNet().to(DEV)
        self.tq.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.LR)
        self.buf = ReplayBuffer()
        self.steps = 0
        self.prev = None                              # (obs, action_idx, levels_completed)
        self.visits = defaultdict(int)
        # read by eval_agents.py (agent_stats) and useful for curves
        self.stats = {"acts": np.zeros(N_ACTIONS, np.int64), "loss": [], "eps": [], "ncand": [], "mem_masked": 0}

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MODE}.{self.ASPACE}.mem{self.MEM}.{self.MAX_ACTIONS}"

    def is_done(self, frames, latest_frame) -> bool:
        return latest_frame.state is GameState.WIN

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _obs(frame: FrameData) -> np.ndarray:
        return np.asarray(frame.frame[-1], dtype=np.uint8)       # (64,64), last layer

    @staticmethod
    def _avail(frame: FrameData) -> np.ndarray:
        """bool[6]: is ACTION1..ACTION6 available in this frame?"""
        av = set(getattr(frame, "available_actions", None) or range(1, 7))
        m = np.array([(i in av) for i in range(1, 7)], bool)
        return m if m.any() else np.ones(6, bool)

    def _reward(self, obs, frame) -> float:
        r = 0.0
        if self.prev is not None and frame.levels_completed > self.prev[2]:
            r += 1.0
        if frame.state is GameState.GAME_OVER:
            r -= 0.1
        if self.NOVELTY:
            key = hash(obs.tobytes())
            self.visits[key] += 1
            r += 0.01 / np.sqrt(self.visits[key])
        return r

    def _mask(self, frame: FrameData, obs: np.ndarray) -> np.ndarray:
        """bool[4101]: valid actions. In 'object' mode clicks are limited to candidates."""
        av = self._avail(frame)
        if av[5]:
            clk = (click_candidates(obs, max_comp=self.MAXCOMP, use_grid=bool(self.GRIDFB))
                   if self.ASPACE == "object" else np.ones(N_CLICK, bool))
            self.stats["ncand"].append(int(clk.sum()))
        else:
            clk = np.zeros(N_CLICK, bool)
        return np.concatenate([av[:N_SIMPLE], clk])

    def _random_idx(self, mask: np.ndarray) -> int:
        """Uniform over valid action TYPES, then uniform over valid click targets.
        (Uniform over all 4101 flat indices would be ~99.9% clicks.)"""
        types = [i for i in range(N_SIMPLE) if mask[i]] + ([N_SIMPLE] if mask[N_SIMPLE:].any() else [])
        t = random.choice(types)
        if t < N_SIMPLE:
            return t
        if self.ASPACE == "pixel" and not self.MEM:
            return N_SIMPLE + random.randrange(N_CLICK)      # identical RNG use to baseline
        return N_SIMPLE + random.choice(np.flatnonzero(mask[N_SIMPLE:]).tolist())

    def _learn(self):
        if self.buf.n < max(self.WARMUP, self.BATCH):
            return
        s, a, r, s2, d, m2 = self.buf.sample(self.BATCH)
        s = torch.as_tensor(s, device=DEV).long()
        s2 = torch.as_tensor(s2, device=DEV).long()
        a = torch.as_tensor(a, device=DEV)
        r = torch.as_tensor(r, device=DEV)
        d = torch.as_tensor(d, device=DEV)
        m2 = torch.as_tensor(m2, device=DEV)
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nxt = self.tq(s2).masked_fill(~m2, -1e9).max(1).values
            tgt = r + self.GAMMA * nxt * (1 - d)
        loss = F.smooth_l1_loss(q, tgt)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.stats["loss"].append(float(loss))

    def _to_action(self, idx: int) -> GameAction:
        if idx < N_SIMPLE:
            act = GameAction[f"ACTION{idx + 1}"]
            act.reasoning = f"{self.MODE} simple {idx + 1}"
        else:
            act = GameAction.ACTION6
            c = idx - N_SIMPLE
            act.set_data({"x": c % GRID, "y": c // GRID})
            act.reasoning = {"why": f"{self.MODE} click"}
        return act

    # ---- main loop ---------------------------------------------------------
    def choose_action(self, frames, latest_frame) -> GameAction:
        # Terminal / start: close the episode, then RESET (weights + buffer persist).
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if self.prev is not None and latest_frame.frame:
                obs = self._obs(latest_frame)
                self.buf.push(self.prev[0], self.prev[1], self._reward(obs, latest_frame),
                              obs, 1.0, np.ones(N_ACTIONS, bool))   # terminal: mask unused
                self.prev = None
            return GameAction.RESET

        obs = self._obs(latest_frame)
        mask = self._mask(latest_frame, obs)
        key = hash(obs.tobytes())

        if self.prev is not None:
            if self.MEM:                               # update the no-op memory
                pk, pa = self.prev[3], self.prev[1]
                self.tried[pk].add(pa)
                if key == pk:
                    self.noop[pk].add(pa)
            self.buf.push(self.prev[0], self.prev[1], self._reward(obs, latest_frame),
                          obs, 0.0, mask)
            if self.MODE == "dqn":
                self._learn()
                if self.steps % self.TARGET_EVERY == 0:
                    self.tq.load_state_dict(self.q.state_dict())

        sel = mask                                      # actions we are willing to take now
        if self.MEM:
            bad = self.noop.get(key)
            if bad:
                cand = mask.copy()
                cand[list(bad)] = False
                if cand.any():                          # else: everything is a known no-op
                    sel = cand
                    self.stats["mem_masked"] += int(mask.sum() - cand.sum())

        frac = min(1.0, self.steps / max(1, self.EPS_DECAY))
        eps = self.EPS_START + (self.EPS_END - self.EPS_START) * frac
        self.stats["eps"].append(eps)

        if self.MODE == "random" or random.random() < eps:
            pool = sel
            if self.MEM:                                # untried-first exploration
                tr = self.tried.get(key)
                if tr:
                    u = sel.copy()
                    u[list(tr)] = False
                    if u.any():
                        pool = u
            idx = self._random_idx(pool)
        else:
            with torch.no_grad():
                qv = self.q(torch.as_tensor(obs, device=DEV).long().unsqueeze(0))[0].cpu().numpy()
            qv[~sel] = -np.inf
            idx = int(qv.argmax())

        self.prev = (obs, idx, latest_frame.levels_completed, key)
        self.stats["acts"][idx] += 1
        self.steps += 1
        return self._to_action(idx)