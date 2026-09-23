# -*- coding: utf-8 -*-
"""
Pilot calibration for the external weighted red2pack reference.

The pilot reproduces 30 representative final states from the large-scale
benchmark and compares strong-CHILS and DRP with 1, 5 and 15 second budgets.
It spans ER, BA and WS graphs at n=2,000 and n=20,000, all three weight modes,
and the four real topologies. The experiment is resumable and writes results
after every completed solve.

The published experiment selected strong-CHILS with strong reductions and a
five-second budget for the full external comparison.

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
from typing import Dict, Iterable, List, Optional, Set, Tuple

# red2pack / CHILS is internally parallel. Avoid running several outer Python
# processes at once; let one solver use the CPU instead.
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
OUTPUT_DIR = BASE_DIR / "red2pack_pilot_output"
RESULTS_CSV = OUTPUT_DIR / "pilot_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "pilot_summary.csv"

# Published external methods chosen for the pilot.
# The 2026 weighted-red2pack paper reports strong-CHILS as the quality-oriented
# recommendation; DRP is included as a complementary native MW2PS heuristic.
SOLVERS = (
    ("R2P_CHILS_STRONG", "chils", "strong"),
    ("R2P_DRP", "drp", ""),
)

TIME_LIMITS_S = (1.0, 5.0, 15.0)

# Pilot coverage: deliberately small enough to finish quickly, but spans graph
# family, scale, topology source, and all weight distributions.
PILOT_SYNTHETIC_N = {2000, 20000}
PILOT_SYNTHETIC_SEED = 3101
PILOT_REAL_SEED = 4101
PILOT_REFERENCE_FRACTION = 1.00

# red2pack uses integer node weights internally. large-scale benchmark has floating weights
# for degree-correlated modes. Encode them at high precision, then evaluate the
# returned vertex set using the ORIGINAL large-scale floating objective.
WEIGHT_SCALE = 10_000

# Numerical tolerance only for comparisons in the original floating objective.
EPS = 1e-9


# ============================================================
# SMALL RED2PACK SELF-TEST
# ============================================================

def red2pack_selftest() -> None:
    g = Graph(num_nodes=5)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        g.add_edge(u, v)
    for v, w in enumerate([10, 20, 30, 40, 50]):
        g.set_node_weight(v, w)
    g.finalize()

    expected = 70
    for alg in ("exact_weighted", "chils", "drp"):
        r = IndependenceProblems.two_packing(
            g,
            algorithm=alg,
            time_limit=2.0,
            seed=0,
        )
        if int(r.weight) != expected:
            raise AssertionError(
                f"red2pack self-test failed for {alg}: "
                f"expected {expected}, got {r.weight}"
            )

    print("SELFTEST: PASS")


# ============================================================
# EXACT STAGE-6 STATE REPRODUCTION
# ============================================================

def choose_pilot_tasks() -> List[bench.Task]:
    # Force the original full large-scale design even if RUN_MODE was later edited.
    bench.RUN_MODE = "full"
    tasks, _ = bench.make_tasks()

    selected: List[bench.Task] = []
    for task in tasks:
        if task.source == "synthetic":
            if (
                task.n in PILOT_SYNTHETIC_N
                and task.seed == PILOT_SYNTHETIC_SEED
            ):
                selected.append(task)
        elif task.source == "real":
            if task.seed == PILOT_REAL_SEED:
                selected.append(task)

    return selected


def get_final_reference_row(
    ref_df: pd.DataFrame,
    task: bench.Task,
) -> pd.Series:
    q = ref_df[
        (ref_df["source"] == task.source)
        & (ref_df["graph_name"] == task.graph_name)
        & (ref_df["seed"] == task.seed)
        & (ref_df["weight_mode"] == task.weight_mode)
    ]

    # Synthetic large-scale tasks with the same graph family/seed/weight mode
    # exist at multiple n values.  Match n explicitly, otherwise sorting by
    # step can accidentally select (for example) the n=20000 row for an
    # n=2000 task.  Real tasks store n=0 in Task and the actual loaded size
    # in reference_points.csv, so n is intentionally not filtered there.
    if task.source == "synthetic":
        q = q[ref_df.loc[q.index, "n"].astype(int) == int(task.n)]

    if q.empty:
        raise RuntimeError(
            f"No large-scale reference row found for {task.name}."
        )

    # large-scale benchmark includes a reference at 100% of the stream.
    q = q.sort_values("step")
    row = q.iloc[-1]

    if int(row["step"]) != int(row["total_steps"]):
        raise RuntimeError(
            f"Final reference row missing for {task.name}: "
            f"step={row['step']}, total_steps={row['total_steps']}"
        )

    return row


def reproduce_final_state(
    task: bench.Task,
    ref_row: pd.Series,
) -> Tuple[object, Dict[int, float], int, int, int]:
    """
    Reproduce the large-scale graph, weights, and exact final dynamic graph state.

    Returns
    -------
    G, w, seed_mix, m0, total_steps
    """
    seed_mix = bench.seed_mix_for_task(task)
    rng = np.random.default_rng(seed_mix)

    G = bench.build_task_graph(task, seed_mix)
    m0 = G.number_of_edges()
    edge_pool = bench.DynamicEdgePool(G.edges())

    # Important: make_weights consumes the same main RNG before the update
    # stream, exactly as in large-scale benchmark.
    w = bench.make_weights(
        G=G,
        mode=task.weight_mode,
        rng=rng,
    )

    total_steps = max(
        1,
        int(math.ceil(task.update_load * m0)),
    )

    for _ in range(total_steps):
        bench.apply_random_update(
            G=G,
            edge_pool=edge_pool,
            rng=rng,
        )

    # Strong consistency checks against the already completed large-scale output.
    expected_m0 = int(ref_row["m0"])
    expected_m = int(ref_row["m"])
    expected_steps = int(ref_row["total_steps"])

    if m0 != expected_m0:
        raise AssertionError(
            f"large-scale reproduction mismatch for {task.name}: "
            f"m0={m0}, expected {expected_m0}"
        )
    if G.number_of_edges() != expected_m:
        raise AssertionError(
            f"large-scale reproduction mismatch for {task.name}: "
            f"final m={G.number_of_edges()}, expected {expected_m}"
        )
    if total_steps != expected_steps:
        raise AssertionError(
            f"large-scale reproduction mismatch for {task.name}: "
            f"steps={total_steps}, expected {expected_steps}"
        )

    return G, w, seed_mix, m0, total_steps


# ============================================================
# NETWORKX -> CHSZLABLIB GRAPH
# ============================================================

def encode_weight(x: float) -> int:
    value = int(round(float(x) * WEIGHT_SCALE))
    return max(1, value)


def to_red2pack_graph(G, w: Dict[int, float]) -> Graph:
    n = G.number_of_nodes()

    # large-scale benchmark uses compact labels 0..n-1 for every graph.
    if set(G.nodes()) != set(range(n)):
        raise AssertionError("Expected compact integer labels 0..n-1.")

    rg = Graph(num_nodes=n)

    for u, v in G.edges():
        rg.add_edge(int(u), int(v))

    for v in range(n):
        rg.set_node_weight(v, encode_weight(w[v]))

    rg.finalize()
    return rg


def original_objective(vertices: Iterable[int], w: Dict[int, float]) -> float:
    return float(sum(w[int(v)] for v in vertices))


# ============================================================
# SOLVER CALLS AND CHECKPOINTING
# ============================================================

def solver_seed(seed_mix: int, step: int, algorithm: str) -> int:
    offset = {
        "chils": 31,
        "drp": 47,
        "htwis": 59,
    }.get(algorithm, 83)

    return int(
        (seed_mix + 900_001 + 1009 * step + offset)
        % (2**31 - 1)
    )


def run_red2pack(
    rg: Graph,
    algorithm: str,
    reduction_style: str,
    time_limit_s: float,
    seed: int,
):
    kwargs = dict(
        g=rg,
        algorithm=algorithm,
        time_limit=float(time_limit_s),
        seed=int(seed),
    )
    if reduction_style:
        kwargs["reduction_style"] = reduction_style

    t0 = time.perf_counter()
    result = IndependenceProblems.two_packing(**kwargs)
    wall = time.perf_counter() - t0

    return result, wall


def load_existing_results() -> pd.DataFrame:
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame()


def done_keys(df: pd.DataFrame) -> Set[Tuple[str, str, int, int, str, str, float]]:
    if df.empty:
        return set()

    ok = df[df["status"] == "ok"]
    return {
        (
            str(r.source),
            str(r.graph_name),
            int(r.n),
            int(r.seed),
            str(r.weight_mode),
            str(r.solver),
            float(r.time_limit_s),
        )
        for r in ok.itertuples(index=False)
    }


def atomic_save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ============================================================
# SUMMARY
# ============================================================

def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    rows: List[dict] = []
    group_cols = [
        "solver",
        "algorithm",
        "reduction_style",
        "time_limit_s",
        "source",
    ]

    for keys, g in ok.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        row["states"] = int(len(g))
        row["mean_wall_time_s"] = float(g["wall_time_s"].mean())
        row["median_wall_time_s"] = float(g["wall_time_s"].median())
        row["mean_ratio_to_extended_BKS"] = float(
            g["R2P_ratio_to_extended_BKS"].mean()
        )
        row["min_ratio_to_extended_BKS"] = float(
            g["R2P_ratio_to_extended_BKS"].min()
        )
        row["mean_ratio_to_old_BKS"] = float(
            g["R2P_ratio_to_old_BKS"].mean()
        )
        row["wins_vs_SAFE005"] = int((g["R2P_minus_SAFE005"] > EPS).sum())
        row["ties_vs_SAFE005"] = int((g["R2P_minus_SAFE005"].abs() <= EPS).sum())
        row["losses_vs_SAFE005"] = int((g["R2P_minus_SAFE005"] < -EPS).sum())
        row["mean_relative_gain_vs_SAFE005_pct"] = float(
            100.0 * (
                g["R2P_value"] / g["SAFE005_value"] - 1.0
            ).mean()
        )
        row["mean_relative_gain_vs_STRONG8_pct"] = float(
            100.0 * (
                g["R2P_value"] / g["STRONG8_value"] - 1.0
            ).mean()
        )
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("DYNAMIC MW2PS — RED2PACK PILOT CALIBRATION")
    print("=" * 88)
    print(f"Python          : {sys.version.split()[0]}")
    print(f"Platform        : {platform.platform()}")
    try:
        print(f"chszlablib      : {metadata.version('chszlablib')}")
    except metadata.PackageNotFoundError:
        print("chszlablib      : unknown")
    print(f"OMP_NUM_THREADS : {os.environ.get('OMP_NUM_THREADS')}")
    print(f"large-scale module  : {Path(bench.__file__).resolve()}")
    print(f"Output          : {OUTPUT_DIR}")
    print()

    print("Available two-packing algorithms:")
    print(IndependenceProblems.TWO_PACKING_ALGORITHMS)
    print()

    red2pack_selftest()

    ref_path = bench.OUTPUT_DIR / "reference_points.csv"
    if not ref_path.exists():
        bundled = BASE_DIR / "results" / "large_scale" / "reference_points.csv"
        if bundled.exists():
            ref_path = bundled
        else:
            raise SystemExit(
                "\nMissing large-scale reference points. Run large_scale_benchmark.py "
                "or keep results/large_scale/reference_points.csv in the repository."
            )

    ref_df = pd.read_csv(ref_path)

    tasks = choose_pilot_tasks()
    if not tasks:
        raise RuntimeError("Pilot task selection is empty.")

    real_names = sorted({
        t.graph_name for t in tasks if t.source == "real"
    })
    if real_names:
        bench.prepare_snap_datasets(real_names)

    existing = load_existing_results()
    completed = done_keys(existing)
    records = existing.to_dict("records") if not existing.empty else []

    total_planned = len(tasks) * len(SOLVERS) * len(TIME_LIMITS_S)
    print()
    print(f"Pilot tasks      : {len(tasks)}")
    print(f"Solvers          : {len(SOLVERS)}")
    print(f"Time limits      : {TIME_LIMITS_S}")
    print(f"Planned solves   : {total_planned}")
    print(f"Already complete : {len(completed)}")
    print()

    solve_counter = len(completed)

    for task_index, task in enumerate(tasks, start=1):
        ref_row = get_final_reference_row(ref_df, task)

        print(
            f"[state {task_index:>2}/{len(tasks)}] "
            f"{task.name} — reproducing final large-scale state...",
            flush=True,
        )

        G, w, seed_mix, m0, total_steps = reproduce_final_state(
            task,
            ref_row,
        )

        rg = to_red2pack_graph(G, w)

        old_bks = float(ref_row["BKS_value"])
        benchmark_values = {
            "LOCAL": float(ref_row["LOCAL_value"]),
            "SAFE005": float(ref_row["SAFE005_value"]),
            "SAFE010": float(ref_row["SAFE010_value"]),
            "STRONG8": float(ref_row["STRONG8_value"]),
        }

        for solver_name, algorithm, reduction_style in SOLVERS:
            seed = solver_seed(seed_mix, total_steps, algorithm)

            for limit in TIME_LIMITS_S:
                key = (
                    task.source,
                    task.graph_name,
                    int(G.number_of_nodes()),
                    int(task.seed),
                    task.weight_mode,
                    solver_name,
                    float(limit),
                )

                if key in completed:
                    print(
                        f"    SKIP {solver_name:18s} t={limit:>4.0f}s",
                        flush=True,
                    )
                    continue

                print(
                    f"    RUN  {solver_name:18s} t={limit:>4.0f}s ... ",
                    end="",
                    flush=True,
                )

                try:
                    result, wall = run_red2pack(
                        rg=rg,
                        algorithm=algorithm,
                        reduction_style=reduction_style,
                        time_limit_s=limit,
                        seed=seed,
                    )

                    vertices = {int(v) for v in result.vertices}
                    feasible = bench.is_feasible_2packing(G, vertices)
                    if not feasible:
                        raise AssertionError(
                            f"{solver_name} returned an infeasible 2-packing."
                        )

                    value = original_objective(vertices, w)
                    extended_bks = max(old_bks, value)

                    row = {
                        "status": "ok",
                        "source": task.source,
                        "graph_name": task.graph_name,
                        "n": int(G.number_of_nodes()),
                        "m0": int(m0),
                        "m": int(G.number_of_edges()),
                        "seed": int(task.seed),
                        "weight_mode": task.weight_mode,
                        "step": int(total_steps),
                        "total_steps": int(total_steps),
                        "solver": solver_name,
                        "algorithm": algorithm,
                        "reduction_style": reduction_style,
                        "time_limit_s": float(limit),
                        "solver_seed": int(seed),
                        "weight_scale": int(WEIGHT_SCALE),
                        "wall_time_s": float(wall),
                        "result_size": int(len(vertices)),
                        "encoded_result_weight": int(result.weight),
                        "R2P_value": float(value),
                        "LOCAL_value": benchmark_values["LOCAL"],
                        "SAFE005_value": benchmark_values["SAFE005"],
                        "SAFE010_value": benchmark_values["SAFE010"],
                        "STRONG8_value": benchmark_values["STRONG8"],
                        "old_BKS_value": float(old_bks),
                        "extended_BKS_value": float(extended_bks),
                        "R2P_ratio_to_old_BKS": (
                            value / old_bks if old_bks > 0 else math.nan
                        ),
                        "R2P_ratio_to_extended_BKS": (
                            value / extended_bks if extended_bks > 0 else math.nan
                        ),
                        "SAFE005_ratio_to_extended_BKS": (
                            benchmark_values["SAFE005"] / extended_bks
                            if extended_bks > 0 else math.nan
                        ),
                        "SAFE010_ratio_to_extended_BKS": (
                            benchmark_values["SAFE010"] / extended_bks
                            if extended_bks > 0 else math.nan
                        ),
                        "STRONG8_ratio_to_extended_BKS": (
                            benchmark_values["STRONG8"] / extended_bks
                            if extended_bks > 0 else math.nan
                        ),
                        "R2P_minus_SAFE005": (
                            value - benchmark_values["SAFE005"]
                        ),
                        "R2P_minus_STRONG8": (
                            value - benchmark_values["STRONG8"]
                        ),
                        "feasible": 1,
                    }

                    records.append(row)
                    current = pd.DataFrame(records)
                    atomic_save(current, RESULTS_CSV)

                    completed.add(key)
                    solve_counter += 1

                    print(
                        f"OK  value={value:.6f}  wall={wall:.3f}s  "
                        f"ratio(old BKS)={row['R2P_ratio_to_old_BKS']:.6f}",
                        flush=True,
                    )

                except Exception as exc:
                    print("FAILED", flush=True)
                    print(
                        f"\nERROR in {task.name}, {solver_name}, "
                        f"time_limit={limit}:\n{type(exc).__name__}: {exc}\n"
                    )
                    raise

        # Free large Python/C++ graph references before constructing next state.
        del rg
        del G

    results = pd.read_csv(RESULTS_CSV)
    summary = build_summary(results)
    atomic_save(summary, SUMMARY_CSV)

    print()
    print("=" * 88)
    print("RED2PACK PILOT COMPLETE")
    print("=" * 88)
    print(f"Results : {RESULTS_CSV}")
    print(f"Summary : {SUMMARY_CSV}")
    print()

    if not summary.empty:
        display_cols = [
            "solver",
            "time_limit_s",
            "source",
            "states",
            "mean_ratio_to_extended_BKS",
            "min_ratio_to_extended_BKS",
            "mean_wall_time_s",
            "wins_vs_SAFE005",
            "ties_vs_SAFE005",
            "losses_vs_SAFE005",
        ]
        print(summary[display_cols].to_string(index=False))
        print()

    print("I will use them to choose the final full external solver(s) and time budget.")


if __name__ == "__main__":
    main()
