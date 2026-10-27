from __future__ import annotations
import random, time
from collections import deque, defaultdict
from typing import Any
import os 

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

N_SIMPLE, GRID = 5, 64
N_CLICK = GRID * GRID
N_ACTIONS = N_SIMPLE + N_CLICK
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _cfg(name: str, default, cast=float):
    return cast(os.environ.get(name, default))

class ReplayBuffer:
    def __init__(self, cap:int = 10000):
        self.cap, self.n, self.i = cap, 0, 0
        self.s = np.zeros((cap, GRID, GRID), np.uint8)
        self.s2 = np.zeros((cap, GRID, GRID), np.uint8)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.m2 = np.zeros((cap, 6), bool) 
    def push(self, s, a, r, s2, d, m2):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i], self.m2[i] = s, a, r, s2, d, m2
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)
    def sample(self, k: int):
        idx = np.random.randint(0, self.n, size=k)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx], self.m2[idx])


class QNet(nn.Module):
    def __init__(self, ch: int=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(16, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
        )
        self.click = nn.Conv2d(ch, 1, 1)          # per-pixel Q for ACTION6
        self.simple = nn.Linear(ch, N_SIMPLE)     # pooled features -> ACTION1-5

    def forward(self, x):                          # x: (B,64,64) long
        x = F.one_hot(x, 16).permute(0, 3, 1, 2).float()
        h = self.conv(x)
        return torch.cat([self.simple(h.mean((2, 3))), self.click(h).flatten(1)], 1)


class MyAgent(Agent):
    MAX_ACTIONS = 500
                                      # "random" for the sanity baseline

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
        self.stats = {"acts": np.zeros(N_ACTIONS, np.int64), "loss": [], "eps": []}

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MODE}.{self.MAX_ACTIONS}"

    def is_done(self, frames, latest_frame) -> bool:
        return latest_frame.state is GameState.WIN

    # ---------- helpers ----------
    @staticmethod
    def _obs(frame: FrameData) -> np.ndarray:
        return np.asarray(frame.frame[-1], dtype=np.int8)   # verify layer layout


    @staticmethod
    def _avail(frame: FrameData) -> np.ndarray:
        """bool[6]: is ACTION1..ACTION6 available in this frame?"""
        av = set(getattr(frame, "available_actions", None) or range(1, 7))
        m = np.array([(i in av) for i in range(1, 7)], bool)
        return m if m.any() else np.ones(6, bool)
    
    def _valid_mask(self, frame) -> np.ndarray:
        m = np.zeros(N_ACTIONS, bool)
        avail = set(getattr(frame, "available_actions", []) or range(1, 7))
        for i in range(1, 6):
            if i in avail: m[i - 1] = True
        if 6 in avail: m[N_SIMPLE:] = True
        return m if m.any() else np.ones(N_ACTIONS, bool)

    def _reward(self, obs, frame) -> float:
        r = 0.0
        if self.prev and frame.levels_completed > self.prev[2]:
            r += 1.0
        if frame.state is GameState.GAME_OVER:
            r -= 0.1
        if self.NOVELTY:
            key = hash(obs.tobytes())
            self.visits[key] += 1
            r += 0.01 / np.sqrt(self.visits[key])
        return r

    def _random_idx(self, av: np.ndarray) -> int:
        """Uniform over available action TYPES, then uniform pixel for clicks.
        (Uniform over all 4101 flat indices would be ~99.9% clicks.)"""
        types = [i for i in range(5) if av[i]] + ([5] if av[5] else [])
        t = random.choice(types)
        return t if t < 5 else N_SIMPLE + random.randrange(N_CLICK)
    
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
            full = torch.cat([m2[:, :N_SIMPLE], m2[:, N_SIMPLE:].expand(-1, N_CLICK)], 1)
            nxt = self.tq(s2).masked_fill(~full, -1e9).max(1).values
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

    # ---------- main ----------
    def choose_action(self, frames, latest_frame) -> GameAction:
        # Terminal / start: close the episode, then RESET (weights + buffer persist).
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if self.prev is not None and latest_frame.frame:
                obs = self._obs(latest_frame)
                self.buf.push(self.prev[0], self.prev[1], self._reward(obs, latest_frame),
                              obs, 1.0, self._avail(latest_frame))
                self.prev = None
            return GameAction.RESET
 
        obs = self._obs(latest_frame)
        av = self._avail(latest_frame)
 
        if self.prev is not None:
            self.buf.push(self.prev[0], self.prev[1], self._reward(obs, latest_frame),
                          obs, 0.0, av)
            if self.MODE == "dqn":
                self._learn()
                if self.steps % self.TARGET_EVERY == 0:
                    self.tq.load_state_dict(self.q.state_dict())
 
        frac = min(1.0, self.steps / max(1, self.EPS_DECAY))
        eps = self.EPS_START + (self.EPS_END - self.EPS_START) * frac
        self.stats["eps"].append(eps)
 
        if self.MODE == "random" or random.random() < eps:
            idx = self._random_idx(av)
        else:
            with torch.no_grad():
                qv = self.q(torch.as_tensor(obs, device=DEV).long().unsqueeze(0))[0].cpu().numpy()
            full = np.concatenate([av[:N_SIMPLE], np.full(N_CLICK, av[5])])
            qv[~full] = -np.inf
            idx = int(qv.argmax())
 
        self.prev = (obs, idx, latest_frame.levels_completed)
        self.stats["acts"][idx] += 1
        self.steps += 1
        return self._to_action(idx)