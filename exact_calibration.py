# -*- coding: utf-8 -*-
"""
Exact-quality calibration for dynamic maximum-weight 2-packing.

This experiment has two parts. First, exact MILP solutions on small static
instances calibrate four static heuristics: WEIGHT, RATIO, STRONG4 and
STRONG8. Second, exact MILP solutions are recomputed along small dynamic edge
streams to measure the quality of local maintenance and periodic refresh.

The implementation is intentionally self-contained so the calibration can be
reproduced independently of the large-scale experiments. It supports a quick
self-test mode, task-level checkpoints, automatic resume and atomic output
writes.

Dependencies: numpy, pandas, networkx, scipy>=1.11.
Tested with Python 3.11 on Windows 11.
"""

from __future__ import annotations

import heapq
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp


# ============================================================
# USER CONFIGURATION
# ============================================================

RUN_MODE = "full"   # "quick" or "full"

# Exact MILPs can themselves use native threads. Four Python workers are safer
# than six on an 8-core desktop.
MAX_WORKERS = 4

OUTPUT_DIR = Path(__file__).resolve().parent / "exact_calibration_output"

AVG_DEGREE = 8
UPDATE_INSERT_PROB = 0.50

# Exact solver time limit for ONE static/dynamic state.
EXACT_TIME_LIMIT_S = 20.0

# Static exact calibration.
FULL_STATIC_N = (60, 80, 100)
FULL_STATIC_SEEDS = (1101, 1102, 1103, 1104)
FULL_GRAPH_TYPES = ("ER", "BA", "WS")
FULL_WEIGHT_MODES = ("uniform", "degree_pos", "degree_neg")

# Dynamic exact calibration.
FULL_DYNAMIC_N = 80
FULL_DYNAMIC_SEEDS = (2101, 2102)
FULL_DYNAMIC_UPDATE_LOAD = 0.05

# Quick environment test.
QUICK_STATIC_N = (40,)
QUICK_STATIC_SEEDS = (1101,)
QUICK_GRAPH_TYPES = ("ER", "BA")
QUICK_WEIGHT_MODES = ("uniform",)
QUICK_DYNAMIC_N = 40
QUICK_DYNAMIC_SEEDS = (2101,)
QUICK_DYNAMIC_UPDATE_LOAD = 0.02

# Strong heuristic parameters.
STRONG4_RESTARTS = 4
STRONG8_RESTARTS = 8
RANDOM_TEMP = 0.18
ONE_SWAP_ROUNDS = 5
ONE_FOR_TWO_ROUNDS = 3
ONE_FOR_TWO_TOPK = 80

# Local candidate budget.
LOCAL_BUDGET = 128

# Strong refresh fractions.
REFRESH_FRACTIONS = {
    "R005": 0.005,
    "R010": 0.010,
}


# ============================================================
# BASIC GRAPH / 2-PACKING UTILITIES
# ============================================================

def n2_set(G: nx.Graph, v: int) -> Set[int]:
    out = {v}
    adj_v = G._adj[v]
    out.update(adj_v.keys())
    for u in adj_v:
        out.update(G._adj[u].keys())
    return out


def is_feasible_2packing(G: nx.Graph, S: Set[int]) -> bool:
    for v in S:
        if ((n2_set(G, v) & S) - {v}):
            return False
    return True


def solution_weight(S: Set[int], w: Dict[int, float]) -> float:
    return float(sum(w[v] for v in S))


def weight_greedy(G: nx.Graph, w: Dict[int, float]) -> Set[int]:
    order = sorted(G.nodes, key=lambda v: (w[v], -v), reverse=True)

    blocked: Set[int] = set()
    S: Set[int] = set()

    for v in order:
        if v not in blocked:
            S.add(v)
            blocked.update(n2_set(G, v))

    return S


# ============================================================
# CONFLICT GRAPH H = G^2 WITHOUT SELF-LOOPS
# ============================================================

def build_conflict_graph(G: nx.Graph) -> Dict[int, Set[int]]:
    """
    H[u] contains all v != u with dist_G(u,v) <= 2.
    A 2-packing in G is exactly an independent set in H.
    """
    H: Dict[int, Set[int]] = {}

    for u in G.nodes:
        seen = set(G._adj[u].keys())

        for x in G._adj[u]:
            seen.update(G._adj[x].keys())

        seen.discard(u)
        H[u] = seen

    return H


# ============================================================
# EXACT MW2PS VIA MILP / HIGHS
# ============================================================

def exact_mw2ps(
    G: nx.Graph,
    w: Dict[int, float],
    time_limit_s: float,
) -> Tuple[float, bool, float, str]:
    """
    MW2PS on G = MWIS on conflict graph H.

    Binary variable x_v.
    For each conflict edge (u,v):
        x_u + x_v <= 1.
    """
    t0 = time.perf_counter()

    nodes = list(G.nodes)
    index = {v: i for i, v in enumerate(nodes)}

    H = build_conflict_graph(G)

    edges: List[Tuple[int, int]] = []

    for u in nodes:
        for v in H[u]:
            if u < v:
                edges.append((u, v))

    if edges:
        rows = np.repeat(np.arange(len(edges), dtype=np.int64), 2)

        cols = np.empty(2 * len(edges), dtype=np.int64)
        data = np.ones(2 * len(edges), dtype=float)

        for r, (u, v) in enumerate(edges):
            cols[2 * r] = index[u]
            cols[2 * r + 1] = index[v]

        A = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(edges), len(nodes)),
        )

        constraints = LinearConstraint(
            A,
            lb=-np.inf * np.ones(len(edges)),
            ub=np.ones(len(edges)),
        )
    else:
        constraints = None

    c = -np.array([w[v] for v in nodes], dtype=float)

    result = milp(
        c=c,
        integrality=np.ones(len(nodes), dtype=np.int8),
        bounds=Bounds(
            np.zeros(len(nodes)),
            np.ones(len(nodes)),
        ),
        constraints=constraints,
        options={
            "time_limit": float(time_limit_s),
            "presolve": True,
        },
    )

    elapsed = time.perf_counter() - t0

    if result.fun is None:
        value = math.nan
    else:
        value = -float(result.fun)

    success = bool(result.success)

    return value, success, elapsed, str(result.message)


# ============================================================
# STRONG STATIC HEURISTIC
# ============================================================

def ratio_greedy_heap(
    H: Dict[int, Set[int]],
    w: Dict[int, float],
    noise: Optional[Dict[int, float]] = None,
) -> Set[int]:
    """
    Residual conflict-degree greedy:
        score(v) = noise(v) * w(v) / (d_remaining(v)+1).

    A lazy heap updates scores when remaining conflict degrees change.
    """
    remaining = set(H.keys())
    degree = {v: len(H[v]) for v in H}
    version = {v: 0 for v in H}

    if noise is None:
        noise = {v: 1.0 for v in H}

    heap: List[Tuple[float, float, int, int]] = []

    for v in H:
        score = noise[v] * w[v] / (degree[v] + 1.0)

        heapq.heappush(
            heap,
            (-score, -w[v], v, 0),
        )

    S: Set[int] = set()

    while remaining:

        while heap:
            _, _, v, ver = heapq.heappop(heap)

            if v in remaining and ver == version[v]:
                break
        else:
            break

        S.add(v)

        to_remove = [v]
        to_remove.extend(
            u for u in H[v]
            if u in remaining
        )

        for r in to_remove:

            if r not in remaining:
                continue

            remaining.remove(r)

            for z in H[r]:

                if z in remaining:

                    degree[z] -= 1
                    version[z] += 1

                    score = (
                        noise[z]
                        * w[z]
                        / (degree[z] + 1.0)
                    )

                    heapq.heappush(
                        heap,
                        (
                            -score,
                            -w[z],
                            z,
                            version[z],
                        ),
                    )

    return S


def one_swap_improve(
    H: Dict[int, Set[int]],
    w: Dict[int, float],
    S: Set[int],
    max_rounds: int,
) -> Set[int]:
    """
    Allow one unselected vertex to replace all currently selected vertices
    conflicting with it when this improves total weight.
    """
    S = set(S)

    for _ in range(max_rounds):

        best_gain = 1e-12
        best_v = None

        for v in H:

            if v in S:
                continue

            conflicts = H[v] & S

            gain = (
                w[v]
                - sum(w[x] for x in conflicts)
            )

            if gain > best_gain:
                best_gain = gain
                best_v = v

        if best_v is None:
            break

        conflicts = H[best_v] & S

        S.difference_update(conflicts)
        S.add(best_v)

        # Greedy augmentation.
        feasible = [
            v
            for v in H
            if v not in S
            and not (H[v] & S)
        ]

        feasible.sort(
            key=lambda v: (
                w[v] / (len(H[v]) + 1.0),
                w[v],
            ),
            reverse=True,
        )

        for v in feasible:
            if not (H[v] & S):
                S.add(v)

    return S


def one_for_two_improve(
    H: Dict[int, Set[int]],
    w: Dict[int, float],
    S: Set[int],
    max_rounds: int,
    topk: int,
) -> Set[int]:
    """
    Allow one selected vertex s to be replaced by one or two mutually
    compatible unselected vertices whose only selected conflict is s.
    """
    S = set(S)

    for _ in range(max_rounds):

        best_gain = 1e-12
        best_s = None
        best_replacement: Optional[Tuple[int, ...]] = None

        buckets = {s: [] for s in S}

        for v in H:

            if v in S:
                continue

            conflicts = H[v] & S

            if len(conflicts) == 1:

                s = next(iter(conflicts))
                buckets[s].append(v)

        for s, candidates in buckets.items():

            candidates.sort(
                key=lambda v: w[v],
                reverse=True,
            )

            candidates = candidates[:topk]

            if candidates:

                gain = w[candidates[0]] - w[s]

                if gain > best_gain:
                    best_gain = gain
                    best_s = s
                    best_replacement = (
                        candidates[0],
                    )

            for i, a in enumerate(candidates):

                for b in candidates[i + 1:]:

                    if b in H[a]:
                        continue

                    gain = (
                        w[a]
                        + w[b]
                        - w[s]
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_s = s
                        best_replacement = (a, b)

        if best_s is None:
            break

        S.remove(best_s)
        S.update(best_replacement)

        # Greedy augmentation.
        feasible = [
            v
            for v in H
            if v not in S
            and not (H[v] & S)
        ]

        feasible.sort(
            key=lambda v: (
                w[v] / (len(H[v]) + 1.0),
                w[v],
            ),
            reverse=True,
        )

        for v in feasible:
            if not (H[v] & S):
                S.add(v)

    return S


def strong_static(
    G: nx.Graph,
    w: Dict[int, float],
    seed: int,
    restarts: int,
) -> Tuple[Set[int], float]:
    """
    Multi-start static heuristic used for quality calibration and refresh.
    """
    t0 = time.perf_counter()

    H = build_conflict_graph(G)

    rng = np.random.default_rng(seed)

    best_S: Optional[Set[int]] = None
    best_value = -math.inf

    for r in range(restarts):

        if r == 0:
            noise = None
        else:
            noise = {
                v: math.exp(
                    RANDOM_TEMP * rng.normal()
                )
                for v in H
            }

        S = ratio_greedy_heap(
            H=H,
            w=w,
            noise=noise,
        )

        S = one_swap_improve(
            H=H,
            w=w,
            S=S,
            max_rounds=ONE_SWAP_ROUNDS,
        )

        S = one_for_two_improve(
            H=H,
            w=w,
            S=S,
            max_rounds=ONE_FOR_TWO_ROUNDS,
            topk=ONE_FOR_TWO_TOPK,
        )

        S = one_swap_improve(
            H=H,
            w=w,
            S=S,
            max_rounds=ONE_SWAP_ROUNDS,
        )

        value = solution_weight(S, w)

        if value > best_value:
            best_value = value
            best_S = set(S)

    elapsed = time.perf_counter() - t0

    assert best_S is not None

    return best_S, elapsed


# ============================================================
# LOCAL DYNAMIC MAINTENANCE
# ============================================================

def new_conflict_after_insertion(
    G: nx.Graph,
    S: Set[int],
    u: int,
    v: int,
) -> Optional[Tuple[int, int]]:

    if u in S:

        if v in S:
            return (u, v)

        for x in G._adj[v]:
            if x != u and x in S:
                return (u, x)

        return None

    if v in S:

        for x in G._adj[u]:
            if x != v and x in S:
                return (v, x)

    return None


def exact_minloss_repair_after_insertion(
    G: nx.Graph,
    w: Dict[int, float],
    S: Set[int],
    u: int,
    v: int,
) -> Tuple[Set[int], Optional[int]]:

    T = set(S)

    conflict = new_conflict_after_insertion(
        G=G,
        S=T,
        u=u,
        v=v,
    )

    if conflict is None:
        return T, None

    a, b = conflict

    if w[a] < w[b]:
        removed = a
    elif w[b] < w[a]:
        removed = b
    else:
        removed = (
            a
            if G.degree[a] >= G.degree[b]
            else b
        )

    T.remove(removed)

    return T, removed


def local_augmentation(
    G: nx.Graph,
    w: Dict[int, float],
    S: Set[int],
    pool: Iterable[int],
    budget: int,
) -> Tuple[Set[int], int]:

    T = set(S)

    candidates = [
        x
        for x in set(pool)
        if x not in T
    ]

    candidates.sort(
        key=lambda x: (w[x], -x),
        reverse=True,
    )

    candidates = candidates[:budget]

    evaluations = 0

    for x in candidates:

        evaluations += 1

        if not (
            (n2_set(G, x) & T)
            - {x}
        ):
            T.add(x)

    return T, evaluations


def local_dynamic_step(
    G: nx.Graph,
    w: Dict[int, float],
    S: Set[int],
    u: int,
    v: int,
    update_type: str,
) -> Tuple[Set[int], int, int]:

    before = set(S)
    T = set(S)

    if update_type == "ins":

        T, removed = (
            exact_minloss_repair_after_insertion(
                G=G,
                w=w,
                S=T,
                u=u,
                v=v,
            )
        )

        if removed is None:
            pool: Set[int] = set()
        else:
            pool = n2_set(G, removed)

    elif update_type == "del":

        pool = {u, v}
        pool.update(G._adj[u].keys())
        pool.update(G._adj[v].keys())

    else:
        raise ValueError(update_type)

    evaluations = 0

    if pool:

        T, evaluations = local_augmentation(
            G=G,
            w=w,
            S=T,
            pool=pool,
            budget=LOCAL_BUDGET,
        )

    recourse = len(
        before.symmetric_difference(T)
    )

    return T, evaluations, recourse


# ============================================================
# GRAPH / WEIGHT GENERATION
# ============================================================

def make_graph(
    graph_type: str,
    n: int,
    seed: int,
) -> nx.Graph:

    if graph_type == "ER":

        p = min(
            AVG_DEGREE / max(1, n - 1),
            0.25,
        )

        return nx.fast_gnp_random_graph(
            n,
            p,
            seed=seed,
        )

    if graph_type == "BA":

        m = max(
            1,
            AVG_DEGREE // 2,
        )

        return nx.barabasi_albert_graph(
            n,
            m,
            seed=seed,
        )

    if graph_type == "WS":

        k = AVG_DEGREE

        if k >= n:
            k = max(
                2,
                n - 1 - ((n - 1) % 2),
            )

        if k % 2 == 1:
            k += 1

        return nx.watts_strogatz_graph(
            n,
            k,
            0.15,
            seed=seed,
        )

    raise ValueError(graph_type)


def make_weights(
    G: nx.Graph,
    mode: str,
    rng: np.random.Generator,
) -> Dict[int, float]:

    nodes = list(G.nodes)

    noise = rng.random(len(nodes))

    if mode == "uniform":

        vals = rng.integers(
            1,
            101,
            size=len(nodes),
        ).astype(float)

    else:

        degree = np.array(
            [
                G.degree[v]
                for v in nodes
            ],
            dtype=float,
        )

        max_degree = max(
            1.0,
            float(degree.max()),
        )

        d = degree / max_degree

        if mode == "degree_pos":
            base = 0.75 * d + 0.25 * noise
        elif mode == "degree_neg":
            base = (
                0.75 * (1.0 - d)
                + 0.25 * noise
            )
        else:
            raise ValueError(mode)

        vals = 1.0 + 99.0 * base

    return {
        v: float(vals[i])
        for i, v in enumerate(nodes)
    }


# ============================================================
# DYNAMIC EDGE POOL
# ============================================================

class DynamicEdgePool:

    def __init__(
        self,
        edges: Iterable[Tuple[int, int]],
    ):

        self.edges: List[Tuple[int, int]] = []
        self.pos: Dict[Tuple[int, int], int] = {}

        for u, v in edges:
            self.add(u, v)

    @staticmethod
    def norm(
        u: int,
        v: int,
    ) -> Tuple[int, int]:

        return (
            (u, v)
            if u < v
            else (v, u)
        )

    def add(
        self,
        u: int,
        v: int,
    ) -> None:

        e = self.norm(u, v)

        if e in self.pos:
            return

        self.pos[e] = len(self.edges)
        self.edges.append(e)

    def remove(
        self,
        u: int,
        v: int,
    ) -> None:

        e = self.norm(u, v)

        idx = self.pos.pop(e)
        last = self.edges.pop()

        if idx < len(self.edges):
            self.edges[idx] = last
            self.pos[last] = idx

    def random_edge(
        self,
        rng: np.random.Generator,
    ) -> Tuple[int, int]:

        idx = int(
            rng.integers(
                0,
                len(self.edges),
            )
        )

        return self.edges[idx]

    def __len__(self) -> int:
        return len(self.edges)


def apply_random_update(
    G: nx.Graph,
    edge_pool: DynamicEdgePool,
    rng: np.random.Generator,
) -> Tuple[str, int, int]:

    n = G.number_of_nodes()

    do_insert = (
        rng.random()
        < UPDATE_INSERT_PROB
        or len(edge_pool) == 0
    )

    if do_insert:

        for _ in range(10000):

            u = int(
                rng.integers(0, n)
            )

            v = int(
                rng.integers(0, n)
            )

            if (
                u != v
                and not G.has_edge(u, v)
            ):

                G.add_edge(u, v)
                edge_pool.add(u, v)

                return "ins", u, v

        do_insert = False

    if not do_insert:

        u, v = edge_pool.random_edge(rng)

        G.remove_edge(u, v)
        edge_pool.remove(u, v)

        return "del", u, v

    raise RuntimeError(
        "Could not generate update."
    )


# ============================================================
# TASKS / CHECKPOINTS
# ============================================================

@dataclass(frozen=True)
class StaticTask:
    graph_type: str
    n: int
    seed: int
    weight_mode: str

    @property
    def name(self) -> str:
        return (
            f"STATIC_{self.graph_type}_n{self.n}_"
            f"s{self.seed}_w{self.weight_mode}"
        )


@dataclass(frozen=True)
class DynamicTask:
    graph_type: str
    n: int
    seed: int
    weight_mode: str
    update_load: float

    @property
    def name(self) -> str:
        return (
            f"DYNAMIC_{self.graph_type}_n{self.n}_"
            f"s{self.seed}_w{self.weight_mode}"
        )


def static_checkpoint(
    task: StaticTask,
) -> Path:

    return (
        OUTPUT_DIR
        / "static_checkpoints"
        / f"{task.name}.csv"
    )


def dynamic_checkpoint(
    task: DynamicTask,
) -> Path:

    return (
        OUTPUT_DIR
        / "dynamic_checkpoints"
        / f"{task.name}.csv"
    )


# ============================================================
# STATIC TASK
# ============================================================

def run_static_task(
    task: StaticTask,
) -> str:

    ckpt = static_checkpoint(task)

    if ckpt.exists():
        return f"SKIP {task.name}"

    seed_mix = (
        task.seed
        + 1000003 * task.n
        + 97 * sum(
            ord(c)
            for c in task.graph_type
        )
        + 193 * sum(
            ord(c)
            for c in task.weight_mode
        )
    ) % (2**32 - 1)

    rng = np.random.default_rng(seed_mix)

    G = make_graph(
        task.graph_type,
        task.n,
        int(seed_mix),
    )

    w = make_weights(
        G,
        task.weight_mode,
        rng,
    )

    exact_value, exact_success, exact_time, exact_msg = (
        exact_mw2ps(
            G=G,
            w=w,
            time_limit_s=EXACT_TIME_LIMIT_S,
        )
    )

    row = {
        "graph_type": task.graph_type,
        "n": task.n,
        "seed": task.seed,
        "weight_mode": task.weight_mode,
        "m": G.number_of_edges(),
        "exact_value": exact_value,
        "exact_success": int(exact_success),
        "exact_time_s": exact_time,
        "exact_message": exact_msg,
    }

    # Old baseline.
    t0 = time.perf_counter()
    S_weight = weight_greedy(G, w)
    t_weight = time.perf_counter() - t0

    # Deterministic ratio greedy.
    t0 = time.perf_counter()
    H = build_conflict_graph(G)
    S_ratio = ratio_greedy_heap(
        H=H,
        w=w,
        noise=None,
    )
    t_ratio = time.perf_counter() - t0

    # STRONG4.
    S4, t4 = strong_static(
        G=G,
        w=w,
        seed=int(seed_mix),
        restarts=STRONG4_RESTARTS,
    )

    # STRONG8.
    S8, t8 = strong_static(
        G=G,
        w=w,
        seed=int(seed_mix) + 12345,
        restarts=STRONG8_RESTARTS,
    )

    methods = {
        "WEIGHT": (
            S_weight,
            t_weight,
        ),
        "RATIO": (
            S_ratio,
            t_ratio,
        ),
        "STRONG4": (
            S4,
            t4,
        ),
        "STRONG8": (
            S8,
            t8,
        ),
    }

    for name, (S, elapsed) in methods.items():

        if not is_feasible_2packing(G, S):
            raise AssertionError(
                f"{task.name}: "
                f"{name} infeasible."
            )

        value = solution_weight(S, w)

        row[f"{name}_value"] = value
        row[f"{name}_time_s"] = elapsed

        if (
            exact_success
            and exact_value > 0
        ):
            row[f"{name}_ratio_to_OPT"] = (
                value / exact_value
            )
        else:
            row[f"{name}_ratio_to_OPT"] = math.nan

    out = pd.DataFrame([row])

    ckpt.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = ckpt.with_suffix(".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, ckpt)

    return f"DONE {task.name}"


# ============================================================
# DYNAMIC TASK
# ============================================================

def run_dynamic_task(
    task: DynamicTask,
) -> str:

    ckpt = dynamic_checkpoint(task)

    if ckpt.exists():
        return f"SKIP {task.name}"

    seed_mix = (
        task.seed
        + 1000003 * task.n
        + 97 * sum(
            ord(c)
            for c in task.graph_type
        )
        + 193 * sum(
            ord(c)
            for c in task.weight_mode
        )
    ) % (2**32 - 1)

    rng = np.random.default_rng(seed_mix)

    G = make_graph(
        task.graph_type,
        task.n,
        int(seed_mix),
    )

    m0 = G.number_of_edges()

    edge_pool = DynamicEdgePool(
        G.edges()
    )

    w = make_weights(
        G,
        task.weight_mode,
        rng,
    )

    S0, init_time = strong_static(
        G=G,
        w=w,
        seed=int(seed_mix),
        restarts=STRONG4_RESTARTS,
    )

    states = {
        "LOCAL": set(S0),
        "R005": set(S0),
        "R010": set(S0),
    }

    refresh_intervals = {
        name: max(
            1,
            int(math.ceil(frac * m0)),
        )
        for name, frac
        in REFRESH_FRACTIONS.items()
    }

    total_steps = max(
        1,
        int(
            math.ceil(
                task.update_load * m0
            )
        ),
    )

    records: List[dict] = []

    for step in range(
        1,
        total_steps + 1,
    ):

        update_type, u, v = (
            apply_random_update(
                G=G,
                edge_pool=edge_pool,
                rng=rng,
            )
        )

        exact_value, exact_success, exact_time, exact_msg = (
            exact_mw2ps(
                G=G,
                w=w,
                time_limit_s=EXACT_TIME_LIMIT_S,
            )
        )

        row = {
            "graph_type": task.graph_type,
            "n": task.n,
            "seed": task.seed,
            "weight_mode": task.weight_mode,
            "step": step,
            "total_steps": total_steps,
            "m0": m0,
            "m": G.number_of_edges(),
            "update_type": update_type,
            "exact_value": exact_value,
            "exact_success": int(exact_success),
            "exact_time_s": exact_time,
            "exact_message": exact_msg,
            "initial_strong_time_s": init_time,
        }

        # LOCAL.
        t0 = time.perf_counter()

        new_state, evals, recourse = (
            local_dynamic_step(
                G=G,
                w=w,
                S=states["LOCAL"],
                u=u,
                v=v,
                update_type=update_type,
            )
        )

        elapsed = (
            time.perf_counter() - t0
        )

        states["LOCAL"] = new_state

        row["LOCAL_time_s"] = elapsed
        row["LOCAL_evals"] = evals
        row["LOCAL_recourse"] = recourse
        row["LOCAL_refresh"] = 0

        # R005 / R010.
        for name in ("R005", "R010"):

            interval = (
                refresh_intervals[name]
            )

            do_refresh = (
                step % interval == 0
            )

            before = set(states[name])

            if do_refresh:

                new_state, elapsed = (
                    strong_static(
                        G=G,
                        w=w,
                        seed=(
                            int(seed_mix)
                            + 100000 * step
                            + (
                                5
                                if name == "R005"
                                else 10
                            )
                        ),
                        restarts=(
                            STRONG4_RESTARTS
                        ),
                    )
                )

                evals = 0

                recourse = len(
                    before.symmetric_difference(
                        new_state
                    )
                )

            else:

                t0 = time.perf_counter()

                (
                    new_state,
                    evals,
                    recourse,
                ) = local_dynamic_step(
                    G=G,
                    w=w,
                    S=before,
                    u=u,
                    v=v,
                    update_type=update_type,
                )

                elapsed = (
                    time.perf_counter()
                    - t0
                )

            states[name] = new_state

            row[f"{name}_time_s"] = elapsed
            row[f"{name}_evals"] = evals
            row[f"{name}_recourse"] = recourse
            row[f"{name}_refresh"] = int(
                do_refresh
            )

        for name, S in states.items():

            if not is_feasible_2packing(
                G,
                S,
            ):
                raise AssertionError(
                    f"{task.name}: "
                    f"step={step}, "
                    f"{name} infeasible."
                )

            value = solution_weight(S, w)

            row[f"{name}_value"] = value

            if (
                exact_success
                and exact_value > 0
            ):
                row[
                    f"{name}_ratio_to_OPT"
                ] = value / exact_value
            else:
                row[
                    f"{name}_ratio_to_OPT"
                ] = math.nan

        records.append(row)

    out = pd.DataFrame(records)

    ckpt.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = ckpt.with_suffix(".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, ckpt)

    return (
        f"DONE {task.name} "
        f"(steps={total_steps})"
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize() -> Tuple[Path, Path, Path, Path]:

    static_files = sorted(
        (
            OUTPUT_DIR
            / "static_checkpoints"
        ).glob("*.csv")
    )

    dynamic_files = sorted(
        (
            OUTPUT_DIR
            / "dynamic_checkpoints"
        ).glob("*.csv")
    )

    if not static_files:
        raise RuntimeError(
            "No static checkpoints."
        )

    if not dynamic_files:
        raise RuntimeError(
            "No dynamic checkpoints."
        )

    static_all = pd.concat(
        [
            pd.read_csv(p)
            for p in static_files
        ],
        ignore_index=True,
    )

    dynamic_all = pd.concat(
        [
            pd.read_csv(p)
            for p in dynamic_files
        ],
        ignore_index=True,
    )

    static_all_path = (
        OUTPUT_DIR
        / "static_all_results.csv"
    )

    dynamic_all_path = (
        OUTPUT_DIR
        / "dynamic_all_results.csv"
    )

    static_all.to_csv(
        static_all_path,
        index=False,
    )

    dynamic_all.to_csv(
        dynamic_all_path,
        index=False,
    )

    # ---------------- STATIC SUMMARY ----------------
    static_rows = []

    for name in (
        "WEIGHT",
        "RATIO",
        "STRONG4",
        "STRONG8",
    ):

        valid = static_all[
            static_all[
                f"{name}_ratio_to_OPT"
            ].notna()
        ]

        ratios = valid[
            f"{name}_ratio_to_OPT"
        ]

        static_rows.append({
            "method": name,
            "instances": int(len(valid)),
            "exact_success_rate": float(
                static_all[
                    "exact_success"
                ].mean()
            ),
            "mean_ratio_to_OPT": float(
                ratios.mean()
            ),
            "median_ratio_to_OPT": float(
                ratios.median()
            ),
            "min_ratio_to_OPT": float(
                ratios.min()
            ),
            "p05_ratio_to_OPT": float(
                ratios.quantile(0.05)
            ),
            "optimal_hit_rate": float(
                np.mean(
                    np.isclose(
                        ratios,
                        1.0,
                        atol=1e-8,
                    )
                )
            ),
            "mean_time_ms": float(
                1000.0
                * static_all[
                    f"{name}_time_s"
                ].mean()
            ),
        })

    static_summary = pd.DataFrame(
        static_rows
    )

    static_summary_path = (
        OUTPUT_DIR
        / "static_summary.csv"
    )

    static_summary.to_csv(
        static_summary_path,
        index=False,
    )

    # ---------------- DYNAMIC SUMMARY ----------------
    dynamic_rows = []

    for name in (
        "LOCAL",
        "R005",
        "R010",
    ):

        valid = dynamic_all[
            dynamic_all[
                f"{name}_ratio_to_OPT"
            ].notna()
        ]

        ratios = valid[
            f"{name}_ratio_to_OPT"
        ]

        dynamic_rows.append({
            "method": name,
            "states": int(len(valid)),
            "exact_success_rate": float(
                dynamic_all[
                    "exact_success"
                ].mean()
            ),
            "mean_ratio_to_OPT": float(
                ratios.mean()
            ),
            "median_ratio_to_OPT": float(
                ratios.median()
            ),
            "min_ratio_to_OPT": float(
                ratios.min()
            ),
            "p01_ratio_to_OPT": float(
                ratios.quantile(0.01)
            ),
            "p05_ratio_to_OPT": float(
                ratios.quantile(0.05)
            ),
            "ratio_ge_095_rate": float(
                np.mean(
                    ratios >= 0.95
                )
            ),
            "ratio_ge_099_rate": float(
                np.mean(
                    ratios >= 0.99
                )
            ),
            "mean_time_ms": float(
                1000.0
                * dynamic_all[
                    f"{name}_time_s"
                ].mean()
            ),
            "mean_recourse": float(
                dynamic_all[
                    f"{name}_recourse"
                ].mean()
            ),
            "refresh_rate": float(
                dynamic_all[
                    f"{name}_refresh"
                ].mean()
            ),
        })

    dynamic_summary = pd.DataFrame(
        dynamic_rows
    )

    dynamic_summary_path = (
        OUTPUT_DIR
        / "dynamic_summary.csv"
    )

    dynamic_summary.to_csv(
        dynamic_summary_path,
        index=False,
    )

    return (
        static_all_path,
        static_summary_path,
        dynamic_all_path,
        dynamic_summary_path,
    )


# ============================================================
# SELF-TEST
# ============================================================

def selftest() -> None:

    if not hasattr(
        scipy.optimize,
        "milp",
    ):
        raise RuntimeError(
            "scipy.optimize.milp is unavailable. "
            "Please install scipy>=1.11."
        )

    G = nx.path_graph(5)

    w = {
        0: 1.0,
        1: 2.0,
        2: 3.0,
        3: 4.0,
        4: 5.0,
    }

    value, success, _, _ = exact_mw2ps(
        G=G,
        w=w,
        time_limit_s=5.0,
    )

    if not success:
        raise AssertionError(
            "Exact MILP self-test failed."
        )

    # On P5, {1,4} is feasible with total weight 7.
    if abs(value - 7.0) > 1e-8:
        raise AssertionError(
            f"Unexpected exact value: {value}"
        )

    S4, _ = strong_static(
        G=G,
        w=w,
        seed=123,
        restarts=STRONG4_RESTARTS,
    )

    if not is_feasible_2packing(
        G,
        S4,
    ):
        raise AssertionError(
            "STRONG4 self-test failed."
        )

    print("SELFTEST: PASS")


# ============================================================
# TASK GENERATION
# ============================================================

def make_tasks():

    if RUN_MODE.lower() == "quick":

        static_n = QUICK_STATIC_N
        static_seeds = QUICK_STATIC_SEEDS
        graph_types = QUICK_GRAPH_TYPES
        weight_modes = QUICK_WEIGHT_MODES

        dynamic_n = QUICK_DYNAMIC_N
        dynamic_seeds = QUICK_DYNAMIC_SEEDS
        update_load = QUICK_DYNAMIC_UPDATE_LOAD

    elif RUN_MODE.lower() == "full":

        static_n = FULL_STATIC_N
        static_seeds = FULL_STATIC_SEEDS
        graph_types = FULL_GRAPH_TYPES
        weight_modes = FULL_WEIGHT_MODES

        dynamic_n = FULL_DYNAMIC_N
        dynamic_seeds = FULL_DYNAMIC_SEEDS
        update_load = FULL_DYNAMIC_UPDATE_LOAD

    else:
        raise ValueError(
            'RUN_MODE must be "quick" or "full".'
        )

    static_tasks = [
        StaticTask(
            graph_type=gt,
            n=n,
            seed=seed,
            weight_mode=wm,
        )
        for gt in graph_types
        for n in static_n
        for seed in static_seeds
        for wm in weight_modes
    ]

    dynamic_tasks = [
        DynamicTask(
            graph_type=gt,
            n=dynamic_n,
            seed=seed,
            weight_mode=wm,
            update_load=update_load,
        )
        for gt in graph_types
        for seed in dynamic_seeds
        for wm in weight_modes
    ]

    return static_tasks, dynamic_tasks


# ============================================================
# MAIN
# ============================================================

def run_task_group(
    tasks,
    checkpoint_fn,
    runner,
    label: str,
) -> None:

    pending = [
        task
        for task in tasks
        if not checkpoint_fn(task).exists()
    ]

    print()
    print(f"{label} tasks total : {len(tasks)}")
    print(
        f"{label} completed   : "
        f"{len(tasks) - len(pending)}"
    )
    print(
        f"{label} pending     : "
        f"{len(pending)}"
    )

    if not pending:
        return

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                runner,
                task,
            ): task
            for task in pending
        }

        done = 0

        for future in as_completed(
            futures
        ):

            task = futures[future]

            try:
                msg = future.result()
                done += 1

                print(
                    f"[{done:>3}/"
                    f"{len(pending)}] "
                    f"{msg}",
                    flush=True,
                )

            except Exception as exc:

                print(
                    f"FAILED "
                    f"{task.name}: "
                    f"{exc}",
                    flush=True,
                )

                raise


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        OUTPUT_DIR
        / "static_checkpoints"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        OUTPUT_DIR
        / "dynamic_checkpoints"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "DYNAMIC MW2PS — EXACT-QUALITY CALIBRATION"
    )
    print("=" * 78)

    print(
        f"Python       : "
        f"{os.sys.version.split()[0]}"
    )
    print(
        f"NetworkX     : "
        f"{nx.__version__}"
    )
    print(
        f"NumPy        : "
        f"{np.__version__}"
    )
    print(
        f"Pandas       : "
        f"{pd.__version__}"
    )
    print(
        f"SciPy        : "
        f"{scipy.__version__}"
    )
    print(
        f"Run mode     : "
        f"{RUN_MODE}"
    )
    print(
        f"Workers      : "
        f"{MAX_WORKERS}"
    )
    print(
        f"Output       : "
        f"{OUTPUT_DIR}"
    )

    selftest()

    static_tasks, dynamic_tasks = (
        make_tasks()
    )

    run_task_group(
        tasks=static_tasks,
        checkpoint_fn=static_checkpoint,
        runner=run_static_task,
        label="STATIC",
    )

    run_task_group(
        tasks=dynamic_tasks,
        checkpoint_fn=dynamic_checkpoint,
        runner=run_dynamic_task,
        label="DYNAMIC",
    )

    (
        static_all_path,
        static_summary_path,
        dynamic_all_path,
        dynamic_summary_path,
    ) = summarize()

    print()
    print("=" * 78)
    print("EXACT CALIBRATION COMPLETE")
    print("=" * 78)

    print(
        f"Static all      : "
        f"{static_all_path}"
    )
    print(
        f"Static summary  : "
        f"{static_summary_path}"
    )
    print(
        f"Dynamic all     : "
        f"{dynamic_all_path}"
    )
    print(
        f"Dynamic summary : "
        f"{dynamic_summary_path}"
    )



if __name__ == "__main__":

    import multiprocessing as mp

    mp.freeze_support()
    main()
