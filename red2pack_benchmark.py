# -*- coding: utf-8 -*-
"""
Full external weighted red2pack benchmark for dynamic MW2PS.

The script replays every sampled state from the large-scale dynamic benchmark
and runs strong-CHILS from scratch with strong reductions and a five-second
budget. It verifies graph-state reproduction, solution feasibility and encoded
integer-weight consistency. The external reference is compared with LOCAL,
SAFE005, SAFE010 and the internal STRONG8 diagnostic at 384 sampled states.

Each successful solve is written immediately, so the benchmark is resumable.
For timing, red2pack graph construction and solver wall time are recorded
separately and in total.

Dependencies: chszlablib, numpy, pandas, networkx.
Tested with chszlablib 0.5.27 under WSL2/Linux.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

# CHILS is internally parallel. Run one external solve at a time and let
# red2pack use the CPU rather than nesting outer Python multiprocessing.
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd
from chszlablib import Graph, IndependenceProblems

try:
    import large_scale_benchmark as bench
except ImportError as exc:
    raise SystemExit(
        "Cannot import large_scale_benchmark.py.\n"
        "Place this file in the same folder as large_scale_benchmark.py.\n"
        f"Original error: {exc}"
    )


# ============================================================
# USER CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "red2pack_benchmark_output"

RESULTS_CSV = OUTPUT_DIR / "red2pack_results.csv"
TASK_SUMMARY_CSV = OUTPUT_DIR / "red2pack_task_summary.csv"
SUMMARY_CSV = OUTPUT_DIR / "red2pack_summary.csv"

ALGORITHM = "chils"
REDUCTION_STYLE = "strong"
TIME_LIMIT_S = 5.0

# Same encoding as the pilot pilot. large-scale degree-correlated weights are
# floating point; red2pack uses integer node weights internally.
WEIGHT_SCALE = 10_000

EPS = 1e-9


# ============================================================
# RED2PACK SELF-TEST
# ============================================================

def red2pack_selftest() -> None:
    g = Graph(num_nodes=5)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        g.add_edge(u, v)

    for v, w in enumerate([10, 20, 30, 40, 50]):
        g.set_node_weight(v, w)

    g.finalize()

    result = IndependenceProblems.two_packing(
        g,
        algorithm="exact_weighted",
        time_limit=2.0,
        seed=0,
    )

    if int(result.weight) != 70:
        raise AssertionError(
            f"red2pack self-test failed: expected 70, got {result.weight}"
        )

    print("SELFTEST: PASS")


# ============================================================
# STAGE-6 TASK / REFERENCE HANDLING
# ============================================================

def full_benchmark_tasks() -> List[bench.Task]:
    # Do not depend on a later manual edit of large-scale RUN_MODE.
    bench.RUN_MODE = "full"
    tasks, _ = bench.make_tasks()
    return tasks


def reference_rows_for_task(
    ref_df: pd.DataFrame,
    task: bench.Task,
) -> pd.DataFrame:
    q = ref_df[
        (ref_df["source"] == task.source)
        & (ref_df["graph_name"] == task.graph_name)
        & (ref_df["seed"] == task.seed)
        & (ref_df["weight_mode"] == task.weight_mode)
    ].copy()

    # Synthetic tasks exist at several graph sizes.
    if task.source == "synthetic":
        q = q[q["n"].astype(int) == int(task.n)]

    if q.empty:
        raise RuntimeError(
            f"No large-scale reference rows found for {task.name}."
        )

    q = q.sort_values("step").reset_index(drop=True)

    if q["step"].duplicated().any():
        raise RuntimeError(
            f"Duplicate reference steps found for {task.name}."
        )

    return q


def task_actual_n(task: bench.Task, rows: pd.DataFrame) -> int:
    ns = sorted(set(int(x) for x in rows["n"].tolist()))
    if len(ns) != 1:
        raise RuntimeError(
            f"Unexpected n values for {task.name}: {ns}"
        )
    return ns[0]


# ============================================================
# NETWORKX -> RED2PACK
# ============================================================

def encode_weight(x: float) -> int:
    value = int(round(float(x) * WEIGHT_SCALE))
    return max(1, value)


def to_red2pack_graph(
    G,
    w: Dict[int, float],
) -> Tuple[Graph, float]:
    t0 = time.perf_counter()

    n = G.number_of_nodes()

    if set(G.nodes()) != set(range(n)):
        raise AssertionError(
            "Expected compact integer node labels 0..n-1."
        )

    rg = Graph(num_nodes=n)

    for u, v in G.edges():
        rg.add_edge(int(u), int(v))

    for v in range(n):
        rg.set_node_weight(v, encode_weight(w[v]))

    rg.finalize()

    build_time = time.perf_counter() - t0
    return rg, build_time


def original_objective(
    vertices: Iterable[int],
    w: Dict[int, float],
) -> float:
    return float(sum(w[int(v)] for v in vertices))


def encoded_objective(
    vertices: Iterable[int],
    w: Dict[int, float],
) -> int:
    return int(sum(encode_weight(w[int(v)]) for v in vertices))


# ============================================================
# SOLVER / CHECKPOINTING
# ============================================================

def solver_seed(
    seed_mix: int,
    step: int,
) -> int:
    return int(
        (seed_mix + 900_001 + 1009 * int(step) + 31)
        % (2**31 - 1)
    )


def run_red2pack(
    rg: Graph,
    seed: int,
):
    t0 = time.perf_counter()

    result = IndependenceProblems.two_packing(
        rg,
        algorithm=ALGORITHM,
        reduction_style=REDUCTION_STYLE,
        time_limit=float(TIME_LIMIT_S),
        seed=int(seed),
    )

    solver_time = time.perf_counter() - t0
    return result, solver_time


def atomic_save(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_existing_results() -> pd.DataFrame:
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame()


def result_key(
    source: str,
    graph_name: str,
    n: int,
    seed: int,
    weight_mode: str,
    step: int,
) -> Tuple[str, str, int, int, str, int]:
    return (
        str(source),
        str(graph_name),
        int(n),
        int(seed),
        str(weight_mode),
        int(step),
    )


def completed_keys(
    df: pd.DataFrame,
) -> Set[Tuple[str, str, int, int, str, int]]:
    if df.empty:
        return set()

    ok = df[df["status"] == "ok"]

    return {
        result_key(
            r.source,
            r.graph_name,
            r.n,
            r.seed,
            r.weight_mode,
            r.step,
        )
        for r in ok.itertuples(index=False)
    }


# ============================================================
# STAGE-6 TASK SUMMARY MATCHING
# ============================================================

def benchmark_task_time_lookup(
    benchmark_task_summary: pd.DataFrame,
    task: bench.Task,
    actual_n: int,
) -> pd.Series:
    q = benchmark_task_summary[
        (benchmark_task_summary["source"] == task.source)
        & (benchmark_task_summary["graph_name"] == task.graph_name)
        & (benchmark_task_summary["seed"] == task.seed)
        & (benchmark_task_summary["weight_mode"] == task.weight_mode)
        & (benchmark_task_summary["n"].astype(int) == int(actual_n))
    ]

    if len(q) != 1:
        raise RuntimeError(
            f"Could not uniquely match large-scale task summary for {task.name}; "
            f"matches={len(q)}."
        )

    return q.iloc[0]


# ============================================================
# SUMMARIES
# ============================================================

QUALITY_METHODS = (
    "LOCAL",
    "SAFE005",
    "SAFE010",
    "STRONG8",
    "R2P",
)


def build_task_summary(
    results: pd.DataFrame,
    benchmark_task_summary: pd.DataFrame,
) -> pd.DataFrame:
    ok = results[results["status"] == "ok"].copy()

    keys = [
        "source",
        "graph_name",
        "n",
        "seed",
        "weight_mode",
    ]

    rows: List[dict] = []

    for group_key, g in ok.groupby(keys, sort=True):
        row = dict(zip(keys, group_key))

        row["reference_points"] = int(len(g))
        row["R2P_mean_build_time_ms"] = float(
            1000.0 * g["R2P_build_time_s"].mean()
        )
        row["R2P_mean_solver_time_ms"] = float(
            1000.0 * g["R2P_solver_time_s"].mean()
        )
        row["R2P_mean_total_time_ms"] = float(
            1000.0 * g["R2P_total_time_s"].mean()
        )

        row["R2P_improves_old_BKS_count"] = int(
            (g["R2P_value"] > g["old_BKS_value"] + EPS).sum()
        )
        row["R2P_ties_old_BKS_count"] = int(
            (
                (g["R2P_value"] - g["old_BKS_value"]).abs()
                <= EPS
            ).sum()
        )
        row["R2P_below_old_BKS_count"] = int(
            (g["R2P_value"] < g["old_BKS_value"] - EPS).sum()
        )

        row["mean_extended_BKS_gain_pct"] = float(
            100.0
            * (
                g["extended_BKS_value"] / g["old_BKS_value"] - 1.0
            ).mean()
        )

        for method in QUALITY_METHODS:
            ratios = g[f"{method}_ratio_to_extended_BKS"]
            row[f"{method}_mean_ratio_to_extended_BKS"] = float(
                ratios.mean()
            )
            row[f"{method}_min_ratio_to_extended_BKS"] = float(
                ratios.min()
            )

        # Recover mean dynamic update times from the original large-scale
        # task_summary.csv, then compare them with the external from-scratch
        # baseline total cost (conversion + solver).
        q = benchmark_task_summary[
            (benchmark_task_summary["source"] == row["source"])
            & (benchmark_task_summary["graph_name"] == row["graph_name"])
            & (benchmark_task_summary["n"].astype(int) == int(row["n"]))
            & (benchmark_task_summary["seed"] == int(row["seed"]))
            & (benchmark_task_summary["weight_mode"] == row["weight_mode"])
        ]

        if len(q) != 1:
            raise RuntimeError(
                "large-scale task summary match failed for "
                f"{row['source']} / {row['graph_name']} / n={row['n']} / "
                f"seed={row['seed']} / {row['weight_mode']}; matches={len(q)}"
            )

        s6row = q.iloc[0]
        r2p_ms = row["R2P_mean_total_time_ms"]

        for method in ("LOCAL", "SAFE005", "SAFE010"):
            dyn_ms = float(s6row[f"{method}_mean_time_ms"])
            row[f"{method}_mean_time_ms"] = dyn_ms
            row[f"{method}_speedup_vs_R2P"] = (
                r2p_ms / dyn_ms
                if dyn_ms > 0
                else math.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_summary(
    results: pd.DataFrame,
    task_summary: pd.DataFrame,
) -> pd.DataFrame:
    ok = results[results["status"] == "ok"].copy()

    rows: List[dict] = []

    # Source-level summary.
    for source, g in ok.groupby("source", sort=True):
        row = {
            "group_type": "source",
            "source": source,
            "graph_name": "ALL",
            "n": -1,
            "states": int(len(g)),
            "tasks": int(
                task_summary[task_summary["source"] == source].shape[0]
            ),
        }

        row["R2P_mean_build_time_ms"] = float(
            1000.0 * g["R2P_build_time_s"].mean()
        )
        row["R2P_mean_solver_time_ms"] = float(
            1000.0 * g["R2P_solver_time_s"].mean()
        )
        row["R2P_mean_total_time_ms"] = float(
            1000.0 * g["R2P_total_time_s"].mean()
        )

        row["R2P_improves_old_BKS_count"] = int(
            (g["R2P_value"] > g["old_BKS_value"] + EPS).sum()
        )
        row["mean_extended_BKS_gain_pct"] = float(
            100.0
            * (
                g["extended_BKS_value"] / g["old_BKS_value"] - 1.0
            ).mean()
        )

        for method in QUALITY_METHODS:
            ratios = g[f"{method}_ratio_to_extended_BKS"]
            row[f"{method}_mean_ratio_to_extended_BKS"] = float(
                ratios.mean()
            )
            row[f"{method}_min_ratio_to_extended_BKS"] = float(
                ratios.min()
            )
            row[f"{method}_p05_ratio_to_extended_BKS"] = float(
                ratios.quantile(0.05)
            )

        ts = task_summary[task_summary["source"] == source]

        for method in ("LOCAL", "SAFE005", "SAFE010"):
            row[f"{method}_mean_speedup_vs_R2P"] = float(
                ts[f"{method}_speedup_vs_R2P"].mean()
            )
            row[f"{method}_median_speedup_vs_R2P"] = float(
                ts[f"{method}_speedup_vs_R2P"].median()
            )

        rows.append(row)

    # Synthetic scaling summary by n.
    syn = ok[ok["source"] == "synthetic"]
    if not syn.empty:
        for n, g in syn.groupby("n", sort=True):
            row = {
                "group_type": "synthetic_n",
                "source": "synthetic",
                "graph_name": "ALL",
                "n": int(n),
                "states": int(len(g)),
                "tasks": int(
                    task_summary[
                        (task_summary["source"] == "synthetic")
                        & (task_summary["n"].astype(int) == int(n))
                    ].shape[0]
                ),
            }

            row["R2P_mean_build_time_ms"] = float(
                1000.0 * g["R2P_build_time_s"].mean()
            )
            row["R2P_mean_solver_time_ms"] = float(
                1000.0 * g["R2P_solver_time_s"].mean()
            )
            row["R2P_mean_total_time_ms"] = float(
                1000.0 * g["R2P_total_time_s"].mean()
            )

            row["R2P_improves_old_BKS_count"] = int(
                (g["R2P_value"] > g["old_BKS_value"] + EPS).sum()
            )
            row["mean_extended_BKS_gain_pct"] = float(
                100.0
                * (
                    g["extended_BKS_value"] / g["old_BKS_value"] - 1.0
                ).mean()
            )

            for method in QUALITY_METHODS:
                ratios = g[f"{method}_ratio_to_extended_BKS"]
                row[f"{method}_mean_ratio_to_extended_BKS"] = float(
                    ratios.mean()
                )
                row[f"{method}_min_ratio_to_extended_BKS"] = float(
                    ratios.min()
                )
                row[f"{method}_p05_ratio_to_extended_BKS"] = float(
                    ratios.quantile(0.05)
                )

            ts = task_summary[
                (task_summary["source"] == "synthetic")
                & (task_summary["n"].astype(int) == int(n))
            ]

            for method in ("LOCAL", "SAFE005", "SAFE010"):
                row[f"{method}_mean_speedup_vs_R2P"] = float(
                    ts[f"{method}_speedup_vs_R2P"].mean()
                )
                row[f"{method}_median_speedup_vs_R2P"] = float(
                    ts[f"{method}_speedup_vs_R2P"].median()
                )

            rows.append(row)

    # Real-network breakdown.
    real = ok[ok["source"] == "real"]
    if not real.empty:
        for graph_name, g in real.groupby("graph_name", sort=True):
            row = {
                "group_type": "real_graph",
                "source": "real",
                "graph_name": graph_name,
                "n": int(g["n"].iloc[0]),
                "states": int(len(g)),
                "tasks": int(
                    task_summary[
                        (task_summary["source"] == "real")
                        & (task_summary["graph_name"] == graph_name)
                    ].shape[0]
                ),
            }

            row["R2P_mean_build_time_ms"] = float(
                1000.0 * g["R2P_build_time_s"].mean()
            )
            row["R2P_mean_solver_time_ms"] = float(
                1000.0 * g["R2P_solver_time_s"].mean()
            )
            row["R2P_mean_total_time_ms"] = float(
                1000.0 * g["R2P_total_time_s"].mean()
            )

            row["R2P_improves_old_BKS_count"] = int(
                (g["R2P_value"] > g["old_BKS_value"] + EPS).sum()
            )
            row["mean_extended_BKS_gain_pct"] = float(
                100.0
                * (
                    g["extended_BKS_value"] / g["old_BKS_value"] - 1.0
                ).mean()
            )

            for method in QUALITY_METHODS:
                ratios = g[f"{method}_ratio_to_extended_BKS"]
                row[f"{method}_mean_ratio_to_extended_BKS"] = float(
                    ratios.mean()
                )
                row[f"{method}_min_ratio_to_extended_BKS"] = float(
                    ratios.min()
                )
                row[f"{method}_p05_ratio_to_extended_BKS"] = float(
                    ratios.quantile(0.05)
                )

            ts = task_summary[
                (task_summary["source"] == "real")
                & (task_summary["graph_name"] == graph_name)
            ]

            for method in ("LOCAL", "SAFE005", "SAFE010"):
                row[f"{method}_mean_speedup_vs_R2P"] = float(
                    ts[f"{method}_speedup_vs_R2P"].mean()
                )
                row[f"{method}_median_speedup_vs_R2P"] = float(
                    ts[f"{method}_speedup_vs_R2P"].median()
                )

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("DYNAMIC MW2PS — FULL EXTERNAL RED2PACK BENCHMARK")
    print("=" * 92)
    print(f"Python          : {sys.version.split()[0]}")
    print(f"Platform        : {platform.platform()}")

    try:
        print(f"chszlablib      : {metadata.version('chszlablib')}")
    except metadata.PackageNotFoundError:
        print("chszlablib      : unknown")

    print(f"OMP_NUM_THREADS : {os.environ.get('OMP_NUM_THREADS')}")
    print(f"Algorithm       : {ALGORITHM}")
    print(f"Reduction       : {REDUCTION_STYLE}")
    print(f"Time limit      : {TIME_LIMIT_S:.1f} s")
    print(f"Weight scale    : {WEIGHT_SCALE}")
    print(f"large-scale module  : {Path(bench.__file__).resolve()}")
    print(f"Output          : {OUTPUT_DIR}")
    print()

    if ALGORITHM not in IndependenceProblems.TWO_PACKING_ALGORITHMS:
        raise RuntimeError(
            f"{ALGORITHM!r} not available. Available algorithms: "
            f"{IndependenceProblems.TWO_PACKING_ALGORITHMS}"
        )

    red2pack_selftest()

    ref_path = bench.OUTPUT_DIR / "reference_points.csv"
    benchmark_task_summary_path = bench.OUTPUT_DIR / "task_summary.csv"

    if not ref_path.exists():
        bundled = BASE_DIR / "results" / "large_scale" / "reference_points.csv"
        if bundled.exists():
            ref_path = bundled
        else:
            raise SystemExit(
                "\nMissing large-scale reference file. Run large_scale_benchmark.py "
                "or keep results/large_scale/reference_points.csv in the repository."
            )

    if not benchmark_task_summary_path.exists():
        bundled = BASE_DIR / "results" / "large_scale" / "task_summary.csv"
        if bundled.exists():
            benchmark_task_summary_path = bundled
        else:
            raise SystemExit(
                "\nMissing large-scale task summary. Run large_scale_benchmark.py "
                "or keep results/large_scale/task_summary.csv in the repository."
            )

    ref_df = pd.read_csv(ref_path)
    benchmark_task_summary = pd.read_csv(benchmark_task_summary_path)

    tasks = full_benchmark_tasks()

    real_names = sorted({
        task.graph_name
        for task in tasks
        if task.source == "real"
    })

    if real_names:
        bench.prepare_snap_datasets(real_names)

    # Pre-validate reference coverage and compute total solves.
    task_refs = {}
    expected_states = 0

    for task in tasks:
        q = reference_rows_for_task(ref_df, task)
        task_refs[task.name] = q
        expected_states += len(q)

    existing = load_existing_results()
    completed = completed_keys(existing)
    records = (
        existing.to_dict("records")
        if not existing.empty
        else []
    )

    print()
    print(f"Dynamic tasks      : {len(tasks)}")
    print(f"Reference states   : {expected_states}")
    print(f"Already complete   : {len(completed)}")
    print(f"Pending solves     : {expected_states - len(completed)}")
    print()

    solved_now = 0

    for task_index, task in enumerate(tasks, start=1):
        ref_rows = task_refs[task.name]
        actual_n = task_actual_n(task, ref_rows)

        ref_by_step = {
            int(row["step"]): row
            for _, row in ref_rows.iterrows()
        }
        ref_steps = sorted(ref_by_step)

        # If every checkpoint for this task is already complete, skip the
        # entire replay rather than rebuilding the graph unnecessarily.
        task_keys = {
            result_key(
                task.source,
                task.graph_name,
                actual_n,
                task.seed,
                task.weight_mode,
                step,
            )
            for step in ref_steps
        }

        if task_keys.issubset(completed):
            print(
                f"[task {task_index:>2}/{len(tasks)}] "
                f"{task.name} — ALL CHECKPOINTS COMPLETE, SKIP",
                flush=True,
            )
            continue

        print(
            f"[task {task_index:>2}/{len(tasks)}] "
            f"{task.name} — replaying stream...",
            flush=True,
        )

        seed_mix = bench.seed_mix_for_task(task)
        rng = np.random.default_rng(seed_mix)

        G = bench.build_task_graph(
            task,
            seed_mix,
        )

        actual_n_runtime = G.number_of_nodes()
        m0 = G.number_of_edges()

        if actual_n_runtime != actual_n:
            raise AssertionError(
                f"{task.name}: n mismatch; reproduced {actual_n_runtime}, "
                f"large-scale reference says {actual_n}."
            )

        edge_pool = bench.DynamicEdgePool(G.edges())

        # Same RNG consumption order as large-scale benchmark.
        w = bench.make_weights(
            G=G,
            mode=task.weight_mode,
            rng=rng,
        )

        total_steps = max(
            1,
            int(math.ceil(task.update_load * m0)),
        )

        expected_total_steps = int(ref_rows["total_steps"].iloc[0])
        expected_m0 = int(ref_rows["m0"].iloc[0])

        if m0 != expected_m0:
            raise AssertionError(
                f"{task.name}: m0 mismatch; reproduced {m0}, "
                f"expected {expected_m0}."
            )

        if total_steps != expected_total_steps:
            raise AssertionError(
                f"{task.name}: total_steps mismatch; reproduced "
                f"{total_steps}, expected {expected_total_steps}."
            )

        for step in range(1, total_steps + 1):
            bench.apply_random_update(
                G=G,
                edge_pool=edge_pool,
                rng=rng,
            )

            if step not in ref_by_step:
                continue

            ref_row = ref_by_step[step]

            expected_m = int(ref_row["m"])
            if G.number_of_edges() != expected_m:
                raise AssertionError(
                    f"{task.name}: step={step}, edge-count mismatch; "
                    f"reproduced {G.number_of_edges()}, expected {expected_m}."
                )

            key = result_key(
                task.source,
                task.graph_name,
                actual_n,
                task.seed,
                task.weight_mode,
                step,
            )

            if key in completed:
                print(
                    f"    step {step:>6}/{total_steps}: SKIP",
                    flush=True,
                )
                continue

            print(
                f"    step {step:>6}/{total_steps}: "
                f"build red2pack graph ... ",
                end="",
                flush=True,
            )

            rg, build_time = to_red2pack_graph(
                G=G,
                w=w,
            )

            seed = solver_seed(
                seed_mix=seed_mix,
                step=step,
            )

            print(
                f"solve {TIME_LIMIT_S:.0f}s ... ",
                end="",
                flush=True,
            )

            result, solver_time = run_red2pack(
                rg=rg,
                seed=seed,
            )

            vertices = {int(v) for v in result.vertices}

            if len(vertices) != len(list(result.vertices)):
                raise AssertionError(
                    f"{task.name}, step={step}: duplicate result vertices."
                )

            if not bench.is_feasible_2packing(G, vertices):
                raise AssertionError(
                    f"{task.name}, step={step}: red2pack returned "
                    "an infeasible 2-packing."
                )

            manual_encoded = encoded_objective(
                vertices,
                w,
            )
            reported_encoded = int(result.weight)

            if manual_encoded != reported_encoded:
                raise AssertionError(
                    f"{task.name}, step={step}: encoded-weight mismatch; "
                    f"manual={manual_encoded}, reported={reported_encoded}."
                )

            r2p_value = original_objective(
                vertices,
                w,
            )

            old_bks = float(ref_row["BKS_value"])
            local_value = float(ref_row["LOCAL_value"])
            safe005_value = float(ref_row["SAFE005_value"])
            safe010_value = float(ref_row["SAFE010_value"])
            strong8_value = float(ref_row["STRONG8_value"])

            extended_bks = max(
                old_bks,
                r2p_value,
            )

            total_time = build_time + solver_time

            row = {
                "status": "ok",
                "source": task.source,
                "graph_name": task.graph_name,
                "n": int(actual_n),
                "m0": int(m0),
                "m": int(G.number_of_edges()),
                "seed": int(task.seed),
                "weight_mode": task.weight_mode,
                "step": int(step),
                "total_steps": int(total_steps),
                "normalized_step": float(step / m0),
                "reference_fraction": float(step / total_steps),
                "algorithm": ALGORITHM,
                "reduction_style": REDUCTION_STYLE,
                "time_limit_s": float(TIME_LIMIT_S),
                "solver_seed": int(seed),
                "weight_scale": int(WEIGHT_SCALE),
                "result_size": int(len(vertices)),
                "encoded_result_weight": int(reported_encoded),
                "R2P_build_time_s": float(build_time),
                "R2P_solver_time_s": float(solver_time),
                "R2P_total_time_s": float(total_time),
                "R2P_value": float(r2p_value),
                "LOCAL_value": local_value,
                "SAFE005_value": safe005_value,
                "SAFE010_value": safe010_value,
                "STRONG8_value": strong8_value,
                "old_BKS_value": old_bks,
                "extended_BKS_value": float(extended_bks),
                "extended_BKS_gain_pct": float(
                    100.0 * (extended_bks / old_bks - 1.0)
                    if old_bks > 0
                    else math.nan
                ),
                "R2P_ratio_to_old_BKS": float(
                    r2p_value / old_bks
                    if old_bks > 0
                    else math.nan
                ),
                "LOCAL_ratio_to_extended_BKS": float(
                    local_value / extended_bks
                    if extended_bks > 0
                    else math.nan
                ),
                "SAFE005_ratio_to_extended_BKS": float(
                    safe005_value / extended_bks
                    if extended_bks > 0
                    else math.nan
                ),
                "SAFE010_ratio_to_extended_BKS": float(
                    safe010_value / extended_bks
                    if extended_bks > 0
                    else math.nan
                ),
                "STRONG8_ratio_to_extended_BKS": float(
                    strong8_value / extended_bks
                    if extended_bks > 0
                    else math.nan
                ),
                "R2P_ratio_to_extended_BKS": float(
                    r2p_value / extended_bks
                    if extended_bks > 0
                    else math.nan
                ),
                "R2P_minus_SAFE005": float(
                    r2p_value - safe005_value
                ),
                "R2P_minus_STRONG8": float(
                    r2p_value - strong8_value
                ),
                "feasible": 1,
                "encoded_weight_verified": 1,
            }

            records.append(row)
            current = pd.DataFrame(records)
            atomic_save(current, RESULTS_CSV)

            completed.add(key)
            solved_now += 1

            print(
                f"OK  value={r2p_value:.6f}  "
                f"build={build_time:.3f}s  "
                f"solve={solver_time:.3f}s  "
                f"gain(old BKS)={row['extended_BKS_gain_pct']:.3f}%",
                flush=True,
            )

            del rg

        del G

    # --------------------------------------------------------
    # Final summaries
    # --------------------------------------------------------

    results = pd.read_csv(RESULTS_CSV)

    if len(results[results["status"] == "ok"]) != expected_states:
        raise RuntimeError(
            "External benchmark ended without all expected successful states: "
            f"successful={len(results[results['status'] == 'ok'])}, "
            f"expected={expected_states}."
        )

    task_summary = build_task_summary(
        results=results,
        benchmark_task_summary=benchmark_task_summary,
    )
    atomic_save(task_summary, TASK_SUMMARY_CSV)

    summary = build_summary(
        results=results,
        task_summary=task_summary,
    )
    atomic_save(summary, SUMMARY_CSV)

    print()
    print("=" * 92)
    print("EXTERNAL RED2PACK BENCHMARK COMPLETE")
    print("=" * 92)
    print(f"Successful states : {expected_states}")
    print(f"Solved this run   : {solved_now}")
    print(f"Results           : {RESULTS_CSV}")
    print(f"Task summary      : {TASK_SUMMARY_CSV}")
    print(f"Summary           : {SUMMARY_CSV}")
    print()

    source_rows = summary[
        summary["group_type"] == "source"
    ].copy()

    display_cols = [
        "source",
        "states",
        "R2P_improves_old_BKS_count",
        "mean_extended_BKS_gain_pct",
        "LOCAL_mean_ratio_to_extended_BKS",
        "SAFE010_mean_ratio_to_extended_BKS",
        "SAFE005_mean_ratio_to_extended_BKS",
        "R2P_mean_ratio_to_extended_BKS",
        "R2P_mean_total_time_ms",
        "SAFE010_mean_speedup_vs_R2P",
        "SAFE005_mean_speedup_vs_R2P",
    ]

    print("Source-level revised benchmark:")
    print(source_rows[display_cols].to_string(index=False))
    print()

    print(f"  1) {SUMMARY_CSV.name}")
    print(f"  2) {TASK_SUMMARY_CSV.name}")
    print(f"  3) {RESULTS_CSV.name}")
    print()


if __name__ == "__main__":
    main()
