# -*- coding: utf-8 -*-
"""
Large-scale benchmark for dynamic maximum-weight 2-packing.

Methods:
    LOCAL    exact immediate repair + budgeted local augmentation
    SAFE005  LOCAL + safe STRONG4 refresh every 0.5% of initial edges
    SAFE010  LOCAL + safe STRONG4 refresh every 1.0% of initial edges

The benchmark uses ER, BA and WS synthetic graphs and four public SNAP
network topologies. Dynamic streams consist of reproducible single-edge
insertions/deletions on a fixed vertex set. STRONG8 is recomputed from scratch
at four sampled checkpoints per task as an internal diagnostic reference.

The script supports automatic SNAP download, task-level checkpoint/resume,
atomic writes and reproducible random seeds.

Dependencies: numpy, pandas, networkx.
The original large-scale run used Python 3.11 on Windows 11.
"""

from __future__ import annotations

import gzip
import heapq
import math
import os
import shutil
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# USER CONFIGURATION
# ============================================================

RUN_MODE = "full"        # "quick" or "full"

# STRONG4/STRONG8 build conflict graphs. Three workers are conservative for
# an 8-core / 32-GB desktop and reduce memory pressure.
MAX_WORKERS = 3

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "large_scale_output"
DATA_DIR = BASE_DIR / "large_scale_data"

AVG_DEGREE = 8
UPDATE_INSERT_PROB = 0.50
LOCAL_BUDGET = 128

# Strong heuristic settings.
STRONG4_RESTARTS = 4
STRONG8_RESTARTS = 8
RANDOM_TEMP = 0.18
ONE_SWAP_ROUNDS = 5
ONE_FOR_TWO_ROUNDS = 3
ONE_FOR_TWO_TOPK = 80

# Final safe-refresh policies.
SAFE_POLICIES = {
    "SAFE005": 0.005,
    "SAFE010": 0.010,
}

# Full benchmark: 5% of initial edge count.
FULL_UPDATE_LOAD = 0.05

FULL_SYNTHETIC_N = (2000, 5000, 10000, 20000)
FULL_SYNTHETIC_SEEDS = (3101, 3102)
FULL_GRAPH_TYPES = ("ER", "BA", "WS")
FULL_WEIGHT_MODES = ("uniform", "degree_pos", "degree_neg")
FULL_REAL_SEEDS = (4101, 4102)

# STRONG8 reference checkpoints as fractions of each update stream.
FULL_REFERENCE_FRACTIONS = (0.25, 0.50, 0.75, 1.00)

# Quick test.
QUICK_UPDATE_LOAD = 0.005
QUICK_SYNTHETIC_N = (500,)
QUICK_SYNTHETIC_SEEDS = (3101,)
QUICK_GRAPH_TYPES = ("ER",)
QUICK_WEIGHT_MODES = ("uniform",)
QUICK_REAL_SEEDS = (4101,)
QUICK_REFERENCE_FRACTIONS = (0.50, 1.00)

# Public SNAP datasets.
SNAP_DATASETS = {
    "ca-GrQc": {
        "url": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
        "directed": False,
    },
    "ca-HepTh": {
        "url": "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
        "directed": False,
    },
    "p2p-Gnutella08": {
        "url": "https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz",
        "directed": True,
    },
    "p2p-Gnutella09": {
        "url": "https://snap.stanford.edu/data/p2p-Gnutella09.txt.gz",
        "directed": True,
    },
}

FULL_REAL_DATASETS = tuple(SNAP_DATASETS.keys())
QUICK_REAL_DATASETS = ("ca-GrQc",)


# ============================================================
# BASIC 2-PACKING UTILITIES
# ============================================================

def n2_set(G: nx.Graph, v: int) -> Set[int]:
    """Closed distance-2 neighborhood N_2[v]."""
    out = {v}
    adj_v = G._adj[v]
    out.update(adj_v.keys())

    for u in adj_v:
        out.update(G._adj[u].keys())

    return out


def is_feasible_2packing(G: nx.Graph, S: Set[int]) -> bool:
    """Exact feasibility audit."""
    for v in S:
        if ((n2_set(G, v) & S) - {v}):
            return False

    return True


def solution_weight(S: Set[int], w: Dict[int, float]) -> float:
    return float(sum(w[v] for v in S))


# ============================================================
# CONFLICT GRAPH AND STRONG STATIC HEURISTIC
# ============================================================

def build_conflict_graph(G: nx.Graph) -> Dict[int, Set[int]]:
    """
    H[u] = vertices at graph distance <= 2 from u, excluding u.
    A 2-packing in G is an independent set in H.
    """
    H: Dict[int, Set[int]] = {}

    for u in G.nodes:
        seen = set(G._adj[u].keys())

        for x in G._adj[u]:
            seen.update(G._adj[x].keys())

        seen.discard(u)
        H[u] = seen

    return H


def ratio_greedy_heap(
    H: Dict[int, Set[int]],
    w: Dict[int, float],
    noise: Optional[Dict[int, float]] = None,
) -> Set[int]:
    """
    Residual conflict-degree greedy:
        score(v) = noise(v) * w(v) / (d_remaining(v) + 1).
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


def greedy_augment_conflict_graph(
    H: Dict[int, Set[int]],
    w: Dict[int, float],
    S: Set[int],
) -> Set[int]:
    S = set(S)

    candidates = [
        v
        for v in H
        if v not in S
        and not (H[v] & S)
    ]

    candidates.sort(
        key=lambda v: (
            w[v] / (len(H[v]) + 1.0),
            w[v],
        ),
        reverse=True,
    )

    for v in candidates:
        if not (H[v] & S):
            S.add(v)

    return S


def one_swap_improve(
    H: Dict[int, Set[int]],
    w: Dict[int, float],
    S: Set[int],
    max_rounds: int,
) -> Set[int]:
    """Replace all selected conflicts of one candidate if objective improves."""
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

        S = greedy_augment_conflict_graph(
            H=H,
            w=w,
            S=S,
        )

    return S


def one_for_two_improve(
    H: Dict[int, Set[int]],
    w: Dict[int, float],
    S: Set[int],
    max_rounds: int,
    topk: int,
) -> Set[int]:
    """
    Replace one selected vertex by one or two mutually compatible candidates.
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

        S = greedy_augment_conflict_graph(
            H=H,
            w=w,
            S=S,
        )

    return S


def strong_static(
    G: nx.Graph,
    w: Dict[int, float],
    seed: int,
    restarts: int,
) -> Tuple[Set[int], float]:
    """Multi-start strong static heuristic."""
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
# EXACT IMMEDIATE REPAIR + LOCAL AUGMENTATION
# ============================================================

def new_conflict_after_insertion(
    G: nx.Graph,
    S: Set[int],
    u: int,
    v: int,
) -> Optional[Tuple[int, int]]:
    """
    S was feasible before insertion of (u,v).
    One edge insertion creates at most one selected-selected 2-packing conflict.
    """
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
) -> Tuple[Set[int], int, int]:

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
    added = 0

    for x in candidates:

        evaluations += 1

        if not (
            (n2_set(G, x) & T)
            - {x}
        ):
            T.add(x)
            added += 1

    return T, evaluations, added


def local_dynamic_step(
    G: nx.Graph,
    w: Dict[int, float],
    S: Set[int],
    u: int,
    v: int,
    update_type: str,
) -> Tuple[Set[int], int, int, int]:

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
    added = 0

    if pool:

        T, evaluations, added = (
            local_augmentation(
                G=G,
                w=w,
                S=T,
                pool=pool,
                budget=LOCAL_BUDGET,
            )
        )

    recourse = len(
        before.symmetric_difference(T)
    )

    return (
        T,
        len(pool),
        evaluations,
        recourse,
    )


def safe_refresh_candidate(
    G: nx.Graph,
    w: Dict[int, float],
    current: Set[int],
    seed: int,
) -> Tuple[Set[int], float, bool, float]:

    candidate, elapsed = strong_static(
        G=G,
        w=w,
        seed=seed,
        restarts=STRONG4_RESTARTS,
    )

    current_value = solution_weight(
        current,
        w,
    )

    candidate_value = solution_weight(
        candidate,
        w,
    )

    if candidate_value > current_value + 1e-12:

        relative_gain = (
            (candidate_value - current_value)
            / current_value
            if current_value > 0
            else math.inf
        )

        return (
            candidate,
            elapsed,
            True,
            relative_gain,
        )

    return (
        set(current),
        elapsed,
        False,
        0.0,
    )


# ============================================================
# GRAPH / WEIGHT GENERATION
# ============================================================

def make_synthetic_graph(
    graph_type: str,
    n: int,
    seed: int,
) -> nx.Graph:

    if graph_type == "ER":

        p = min(
            AVG_DEGREE
            / max(1, n - 1),
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
            base = (
                0.75 * d
                + 0.25 * noise
            )

        elif mode == "degree_neg":
            base = (
                0.75 * (1.0 - d)
                + 0.25 * noise
            )

        else:
            raise ValueError(mode)

        vals = (
            1.0
            + 99.0 * base
        )

    return {
        v: float(vals[i])
        for i, v in enumerate(nodes)
    }


# ============================================================
# SNAP DOWNLOAD / LOAD
# ============================================================

def dataset_local_path(
    name: str,
) -> Path:

    filename = (
        SNAP_DATASETS[name]["url"]
        .split("/")[-1]
    )

    return DATA_DIR / filename


def prepare_snap_datasets(
    names: Sequence[str],
) -> None:

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name in names:

        path = dataset_local_path(name)

        if path.exists():
            print(
                f"[data] found {name}: {path.name}"
            )
            continue

        url = SNAP_DATASETS[name]["url"]

        print(
            f"[data] downloading {name} ..."
        )

        tmp = path.with_suffix(
            path.suffix + ".part"
        )

        try:
            with urllib.request.urlopen(
                url,
                timeout=120,
            ) as response, open(
                tmp,
                "wb",
            ) as fout:

                shutil.copyfileobj(
                    response,
                    fout,
                )

            os.replace(
                tmp,
                path,
            )

        except Exception as exc:

            if tmp.exists():
                tmp.unlink()

            raise RuntimeError(
                f"Failed to download {name}.\n"
                f"URL: {url}\n"
                f"Expected local file: {path}\n"
                f"Error: {exc}\n"
                f"You may manually download the .txt.gz file "
                f"and place it in {DATA_DIR}."
            )

        print(
            f"[data] downloaded {name}: {path.name}"
        )


def load_snap_graph(
    name: str,
) -> nx.Graph:

    path = dataset_local_path(name)

    if not path.exists():
        raise FileNotFoundError(path)

    G = nx.Graph()

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            u = int(parts[0])
            v = int(parts[1])

            if u != v:
                G.add_edge(u, v)

    G.remove_edges_from(
        nx.selfloop_edges(G)
    )

    # Compact integer labels improve memory locality.
    G = nx.convert_node_labels_to_integers(
        G,
        ordering="sorted",
    )

    return G


# ============================================================
# DYNAMIC EDGE POOL
# ============================================================

class DynamicEdgePool:

    def __init__(
        self,
        edges: Iterable[Tuple[int, int]],
    ):
        self.edges: List[Tuple[int, int]] = []
        self.pos: Dict[
            Tuple[int, int],
            int,
        ] = {}

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

        for _ in range(20000):

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

        u, v = edge_pool.random_edge(
            rng
        )

        G.remove_edge(u, v)
        edge_pool.remove(u, v)

        return "del", u, v

    raise RuntimeError(
        "Could not generate an edge update."
    )


# ============================================================
# TASKS
# ============================================================

@dataclass(frozen=True)
class Task:
    source: str
    graph_name: str
    n: int
    seed: int
    weight_mode: str
    update_load: float
    reference_fractions: Tuple[float, ...]

    @property
    def name(self) -> str:

        src = (
            "SYN"
            if self.source == "synthetic"
            else "REAL"
        )

        return (
            f"{src}_{self.graph_name}_"
            f"n{self.n}_s{self.seed}_"
            f"w{self.weight_mode}"
        )


def task_checkpoint(
    task: Task,
) -> Path:

    return (
        OUTPUT_DIR
        / "checkpoints"
        / f"{task.name}.csv"
    )


def seed_mix_for_task(
    task: Task,
) -> int:

    text = (
        task.source
        + task.graph_name
        + task.weight_mode
    )

    return int(
        (
            task.seed
            + 1000003 * max(
                1,
                task.n,
            )
            + 97 * sum(
                ord(c)
                for c in text
            )
        )
        % (2**32 - 1)
    )


def build_task_graph(
    task: Task,
    seed_mix: int,
) -> nx.Graph:

    if task.source == "synthetic":

        return make_synthetic_graph(
            graph_type=task.graph_name,
            n=task.n,
            seed=seed_mix,
        )

    if task.source == "real":

        return load_snap_graph(
            task.graph_name
        )

    raise ValueError(task.source)


# ============================================================
# ONE FINAL-BENCHMARK TASK
# ============================================================

def run_task(
    task: Task,
) -> str:

    ckpt = task_checkpoint(task)

    if ckpt.exists():
        return f"SKIP {task.name}"

    seed_mix = seed_mix_for_task(task)

    rng = np.random.default_rng(
        seed_mix
    )

    G = build_task_graph(
        task,
        seed_mix,
    )

    actual_n = G.number_of_nodes()
    m0 = G.number_of_edges()

    edge_pool = DynamicEdgePool(
        G.edges()
    )

    w = make_weights(
        G=G,
        mode=task.weight_mode,
        rng=rng,
    )

    # High-quality common initial state.
    S0, init_time = strong_static(
        G=G,
        w=w,
        seed=seed_mix + 17,
        restarts=STRONG4_RESTARTS,
    )

    if not is_feasible_2packing(
        G,
        S0,
    ):
        raise AssertionError(
            f"{task.name}: initial state infeasible."
        )

    states = {
        "LOCAL": set(S0),
        "SAFE005": set(S0),
        "SAFE010": set(S0),
    }

    refresh_intervals = {
        name: max(
            1,
            int(
                math.ceil(
                    frac * m0
                )
            ),
        )
        for name, frac
        in SAFE_POLICIES.items()
    }

    total_steps = max(
        1,
        int(
            math.ceil(
                task.update_load * m0
            )
        ),
    )

    reference_steps = {
        max(
            1,
            min(
                total_steps,
                int(
                    round(
                        frac
                        * total_steps
                    )
                ),
            ),
        )
        for frac in task.reference_fractions
    }

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

        row = {
            "source": task.source,
            "graph_name": task.graph_name,
            "n": actual_n,
            "m0": m0,
            "m": G.number_of_edges(),
            "seed": task.seed,
            "weight_mode": task.weight_mode,
            "step": step,
            "total_steps": total_steps,
            "normalized_step": step / m0,
            "update_type": update_type,
            "initial_STRONG4_time_s": init_time,
            "is_reference_step": int(
                step in reference_steps
            ),
        }

        # ----------------------------------------------------
        # LOCAL
        # ----------------------------------------------------
        before = set(
            states["LOCAL"]
        )

        t0 = time.perf_counter()

        (
            new_state,
            pool_size,
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

        states["LOCAL"] = new_state

        row["LOCAL_time_s"] = elapsed
        row["LOCAL_pool"] = pool_size
        row["LOCAL_evals"] = evals
        row["LOCAL_recourse"] = recourse
        row["LOCAL_value"] = (
            solution_weight(
                new_state,
                w,
            )
        )
        row["LOCAL_refresh_attempt"] = 0
        row["LOCAL_refresh_accept"] = 0
        row["LOCAL_refresh_gain"] = 0.0

        # ----------------------------------------------------
        # SAFE005 / SAFE010
        # ----------------------------------------------------
        for name in (
            "SAFE005",
            "SAFE010",
        ):

            before = set(
                states[name]
            )

            t0 = time.perf_counter()

            (
                maintained,
                pool_size,
                evals,
                _,
            ) = local_dynamic_step(
                G=G,
                w=w,
                S=before,
                u=u,
                v=v,
                update_type=update_type,
            )

            local_elapsed = (
                time.perf_counter()
                - t0
            )

            attempt = (
                step
                % refresh_intervals[name]
                == 0
            )

            accepted = False
            gain = 0.0
            refresh_elapsed = 0.0
            chosen = maintained

            if attempt:

                (
                    chosen,
                    refresh_elapsed,
                    accepted,
                    gain,
                ) = safe_refresh_candidate(
                    G=G,
                    w=w,
                    current=maintained,
                    seed=(
                        seed_mix
                        + 100000 * step
                        + (
                            5
                            if name == "SAFE005"
                            else 10
                        )
                    ),
                )

            elapsed = (
                local_elapsed
                + refresh_elapsed
            )

            recourse = len(
                before.symmetric_difference(
                    chosen
                )
            )

            states[name] = chosen

            row[f"{name}_time_s"] = elapsed
            row[f"{name}_pool"] = pool_size
            row[f"{name}_evals"] = evals
            row[f"{name}_recourse"] = recourse
            row[f"{name}_value"] = (
                solution_weight(
                    chosen,
                    w,
                )
            )
            row[
                f"{name}_refresh_attempt"
            ] = int(attempt)
            row[
                f"{name}_refresh_accept"
            ] = int(accepted)
            row[
                f"{name}_refresh_gain"
            ] = gain

        # ----------------------------------------------------
        # STRONG8 sampled reference and BKS
        # ----------------------------------------------------
        if step in reference_steps:

            ref_state, ref_time = (
                strong_static(
                    G=G,
                    w=w,
                    seed=(
                        seed_mix
                        + 700000
                        + step
                    ),
                    restarts=(
                        STRONG8_RESTARTS
                    ),
                )
            )

            if not is_feasible_2packing(
                G,
                ref_state,
            ):
                raise AssertionError(
                    f"{task.name}: STRONG8 reference infeasible."
                )

            ref_value = (
                solution_weight(
                    ref_state,
                    w,
                )
            )

            current_values = {
                name: row[f"{name}_value"]
                for name in (
                    "LOCAL",
                    "SAFE005",
                    "SAFE010",
                )
            }

            bks_value = max(
                [ref_value]
                + list(
                    current_values.values()
                )
            )

            row["STRONG8_value"] = (
                ref_value
            )

            row["STRONG8_time_s"] = (
                ref_time
            )

            row["BKS_value"] = (
                bks_value
            )

            for name in (
                "LOCAL",
                "SAFE005",
                "SAFE010",
            ):

                row[
                    f"{name}_ratio_to_STRONG8"
                ] = (
                    current_values[name]
                    / ref_value
                    if ref_value > 0
                    else math.nan
                )

                row[
                    f"{name}_ratio_to_BKS"
                ] = (
                    current_values[name]
                    / bks_value
                    if bks_value > 0
                    else math.nan
                )

            row[
                "STRONG8_ratio_to_BKS"
            ] = (
                ref_value / bks_value
                if bks_value > 0
                else math.nan
            )

        else:

            row["STRONG8_value"] = math.nan
            row["STRONG8_time_s"] = math.nan
            row["BKS_value"] = math.nan

            for name in (
                "LOCAL",
                "SAFE005",
                "SAFE010",
            ):

                row[
                    f"{name}_ratio_to_STRONG8"
                ] = math.nan

                row[
                    f"{name}_ratio_to_BKS"
                ] = math.nan

            row[
                "STRONG8_ratio_to_BKS"
            ] = math.nan

        # ----------------------------------------------------
        # Periodic feasibility audits
        # ----------------------------------------------------
        if (
            step == 1
            or step == total_steps
            or step in reference_steps
            or step % 250 == 0
        ):

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

        records.append(row)

    out = pd.DataFrame(
        records
    )

    ckpt.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = ckpt.with_suffix(
        ".tmp"
    )

    out.to_csv(
        tmp,
        index=False,
    )

    os.replace(
        tmp,
        ckpt,
    )

    return (
        f"DONE {task.name} "
        f"(actual_n={actual_n}, "
        f"m0={m0}, "
        f"steps={total_steps})"
    )


# ============================================================
# SUMMARIES
# ============================================================

def build_summaries() -> Tuple[Path, Path, Path, Path]:

    files = sorted(
        (
            OUTPUT_DIR
            / "checkpoints"
        ).glob("*.csv")
    )

    if not files:
        raise RuntimeError(
            "No large-scale checkpoint files found."
        )

    frames = [
        pd.read_csv(p)
        for p in files
    ]

    all_df = pd.concat(
        frames,
        ignore_index=True,
    )

    all_path = (
        OUTPUT_DIR
        / "all_results.csv"
    )

    all_df.to_csv(
        all_path,
        index=False,
    )

    methods = (
        "LOCAL",
        "SAFE005",
        "SAFE010",
    )

    # --------------------------------------------------------
    # Task-level summary
    # --------------------------------------------------------
    task_rows = []

    task_keys = [
        "source",
        "graph_name",
        "n",
        "m0",
        "seed",
        "weight_mode",
        "total_steps",
    ]

    for keys, g in all_df.groupby(
        task_keys,
        sort=True,
    ):

        row = dict(
            zip(
                task_keys,
                keys,
            )
        )

        for name in methods:

            row[
                f"{name}_mean_time_ms"
            ] = float(
                1000.0
                * g[
                    f"{name}_time_s"
                ].mean()
            )

            row[
                f"{name}_median_time_ms"
            ] = float(
                1000.0
                * g[
                    f"{name}_time_s"
                ].median()
            )

            row[
                f"{name}_mean_recourse"
            ] = float(
                g[
                    f"{name}_recourse"
                ].mean()
            )

            row[
                f"{name}_mean_pool"
            ] = float(
                g[
                    f"{name}_pool"
                ].mean()
            )

            row[
                f"{name}_refresh_attempts"
            ] = int(
                g[
                    f"{name}_refresh_attempt"
                ].sum()
            )

            row[
                f"{name}_refresh_accepts"
            ] = int(
                g[
                    f"{name}_refresh_accept"
                ].sum()
            )

            attempts = row[
                f"{name}_refresh_attempts"
            ]

            accepts = row[
                f"{name}_refresh_accepts"
            ]

            row[
                f"{name}_refresh_accept_rate"
            ] = (
                accepts / attempts
                if attempts > 0
                else 0.0
            )

        ref = g[
            g[
                "is_reference_step"
            ] == 1
        ]

        row["reference_points"] = int(
            len(ref)
        )

        row[
            "STRONG8_mean_time_ms"
        ] = float(
            1000.0
            * ref[
                "STRONG8_time_s"
            ].mean()
        )

        row[
            "STRONG8_mean_ratio_to_BKS"
        ] = float(
            ref[
                "STRONG8_ratio_to_BKS"
            ].mean()
        )

        for name in methods:

            ratios = ref[
                f"{name}_ratio_to_BKS"
            ]

            row[
                f"{name}_mean_ratio_to_BKS"
            ] = float(
                ratios.mean()
            )

            row[
                f"{name}_min_ratio_to_BKS"
            ] = float(
                ratios.min()
            )

            row[
                f"{name}_p05_ratio_to_BKS"
            ] = float(
                ratios.quantile(0.05)
            )

            dyn_ms = row[
                f"{name}_mean_time_ms"
            ]

            ref_ms = row[
                "STRONG8_mean_time_ms"
            ]

            row[
                f"{name}_speedup_vs_STRONG8"
            ] = (
                ref_ms / dyn_ms
                if dyn_ms > 0
                else math.nan
            )

        task_rows.append(row)

    task_summary = pd.DataFrame(
        task_rows
    )

    task_summary_path = (
        OUTPUT_DIR
        / "task_summary.csv"
    )

    task_summary.to_csv(
        task_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Aggregated dynamic summary
    # --------------------------------------------------------
    summary_rows = []

    group_keys = [
        "source",
        "graph_name",
        "n",
        "weight_mode",
    ]

    for keys, g in all_df.groupby(
        group_keys,
        sort=True,
    ):

        row = dict(
            zip(
                group_keys,
                keys,
            )
        )

        row["tasks"] = int(
            g["seed"].nunique()
        )

        row["updates"] = int(
            len(g)
        )

        row["mean_m0"] = float(
            g["m0"].mean()
        )

        for name in methods:

            row[
                f"{name}_mean_time_ms"
            ] = float(
                1000.0
                * g[
                    f"{name}_time_s"
                ].mean()
            )

            row[
                f"{name}_mean_recourse"
            ] = float(
                g[
                    f"{name}_recourse"
                ].mean()
            )

            row[
                f"{name}_mean_pool"
            ] = float(
                g[
                    f"{name}_pool"
                ].mean()
            )

            attempts = int(
                g[
                    f"{name}_refresh_attempt"
                ].sum()
            )

            accepts = int(
                g[
                    f"{name}_refresh_accept"
                ].sum()
            )

            row[
                f"{name}_refresh_accept_rate"
            ] = (
                accepts / attempts
                if attempts > 0
                else 0.0
            )

        ref = g[
            g[
                "is_reference_step"
            ] == 1
        ]

        row[
            "STRONG8_mean_time_ms"
        ] = float(
            1000.0
            * ref[
                "STRONG8_time_s"
            ].mean()
        )

        for name in methods:

            ratios = ref[
                f"{name}_ratio_to_BKS"
            ]

            row[
                f"{name}_mean_ratio_to_BKS"
            ] = float(
                ratios.mean()
            )

            row[
                f"{name}_min_ratio_to_BKS"
            ] = float(
                ratios.min()
            )

            row[
                f"{name}_p05_ratio_to_BKS"
            ] = float(
                ratios.quantile(0.05)
            )

            dyn_ms = row[
                f"{name}_mean_time_ms"
            ]

            ref_ms = row[
                "STRONG8_mean_time_ms"
            ]

            row[
                f"{name}_speedup_vs_STRONG8"
            ] = (
                ref_ms / dyn_ms
                if dyn_ms > 0
                else math.nan
            )

        summary_rows.append(row)

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Reference-only table
    # --------------------------------------------------------
    ref_df = all_df[
        all_df[
            "is_reference_step"
        ] == 1
    ].copy()

    reference_path = (
        OUTPUT_DIR
        / "reference_points.csv"
    )

    ref_df.to_csv(
        reference_path,
        index=False,
    )

    return (
        all_path,
        task_summary_path,
        summary_path,
        reference_path,
    )


# ============================================================
# SELF-TEST
# ============================================================

def selftest() -> None:

    G = nx.path_graph(6)

    w = {
        0: 1.0,
        1: 2.0,
        2: 3.0,
        3: 4.0,
        4: 5.0,
        5: 6.0,
    }

    S4, _ = strong_static(
        G=G,
        w=w,
        seed=123,
        restarts=STRONG4_RESTARTS,
    )

    S8, _ = strong_static(
        G=G,
        w=w,
        seed=456,
        restarts=STRONG8_RESTARTS,
    )

    if not is_feasible_2packing(
        G,
        S4,
    ):
        raise AssertionError(
            "STRONG4 self-test failed."
        )

    if not is_feasible_2packing(
        G,
        S8,
    ):
        raise AssertionError(
            "STRONG8 self-test failed."
        )

    chosen, _, _, _ = (
        safe_refresh_candidate(
            G=G,
            w=w,
            current=S4,
            seed=789,
        )
    )

    if (
        solution_weight(
            chosen,
            w,
        )
        + 1e-12
        < solution_weight(
            S4,
            w,
        )
    ):
        raise AssertionError(
            "Safe refresh degraded the objective."
        )

    print("SELFTEST: PASS")


# ============================================================
# TASK GENERATION
# ============================================================

def make_tasks() -> Tuple[
    List[Task],
    Tuple[str, ...],
]:

    if RUN_MODE.lower() == "quick":

        update_load = QUICK_UPDATE_LOAD
        synthetic_n = QUICK_SYNTHETIC_N
        synthetic_seeds = QUICK_SYNTHETIC_SEEDS
        graph_types = QUICK_GRAPH_TYPES
        weight_modes = QUICK_WEIGHT_MODES
        real_names = QUICK_REAL_DATASETS
        real_seeds = QUICK_REAL_SEEDS
        ref_fractions = QUICK_REFERENCE_FRACTIONS

    elif RUN_MODE.lower() == "full":

        update_load = FULL_UPDATE_LOAD
        synthetic_n = FULL_SYNTHETIC_N
        synthetic_seeds = FULL_SYNTHETIC_SEEDS
        graph_types = FULL_GRAPH_TYPES
        weight_modes = FULL_WEIGHT_MODES
        real_names = FULL_REAL_DATASETS
        real_seeds = FULL_REAL_SEEDS
        ref_fractions = FULL_REFERENCE_FRACTIONS

    else:
        raise ValueError(
            'RUN_MODE must be "quick" or "full".'
        )

    tasks: List[Task] = []

    for gt in graph_types:

        for n in synthetic_n:

            for seed in synthetic_seeds:

                for wm in weight_modes:

                    tasks.append(
                        Task(
                            source="synthetic",
                            graph_name=gt,
                            n=n,
                            seed=seed,
                            weight_mode=wm,
                            update_load=update_load,
                            reference_fractions=tuple(
                                ref_fractions
                            ),
                        )
                    )

    for name in real_names:

        # n=0 because actual size is loaded from the SNAP file.
        for seed in real_seeds:

            for wm in weight_modes:

                tasks.append(
                    Task(
                        source="real",
                        graph_name=name,
                        n=0,
                        seed=seed,
                        weight_mode=wm,
                        update_load=update_load,
                        reference_fractions=tuple(
                            ref_fractions
                        ),
                    )
                )

    return (
        tasks,
        tuple(real_names),
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        OUTPUT_DIR
        / "checkpoints"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 82)
    print(
        "DYNAMIC MW2PS — LARGE-SCALE BENCHMARK"
    )
    print("=" * 82)

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
    print(
        f"Data         : "
        f"{DATA_DIR}"
    )
    print()

    selftest()

    tasks, real_names = (
        make_tasks()
    )

    if real_names:
        prepare_snap_datasets(
            real_names
        )

    pending = [
        task
        for task in tasks
        if not task_checkpoint(
            task
        ).exists()
    ]

    print()
    print(
        f"Total tasks  : "
        f"{len(tasks)}"
    )
    print(
        f"Completed    : "
        f"{len(tasks) - len(pending)}"
    )
    print(
        f"Pending      : "
        f"{len(pending)}"
    )
    print()

    if pending:

        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            futures = {
                executor.submit(
                    run_task,
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

    (
        all_path,
        task_summary_path,
        summary_path,
        reference_path,
    ) = build_summaries()

    print()
    print("=" * 82)
    print("LARGE-SCALE BENCHMARK COMPLETE")
    print("=" * 82)

    print(
        f"All results       : "
        f"{all_path}"
    )
    print(
        f"Task summary      : "
        f"{task_summary_path}"
    )
    print(
        f"Summary           : "
        f"{summary_path}"
    )
    print(
        f"Reference points  : "
        f"{reference_path}"
    )

    print()


if __name__ == "__main__":

    import multiprocessing as mp

    mp.freeze_support()
    main()
