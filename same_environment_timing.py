# -*- coding: utf-8 -*-
"""
Same-environment timing and cross-platform reproduction audit.

The dynamic methods are rerun sequentially under the same WSL2/Linux
environment used for the external red2pack experiment. This avoids comparing
wall-clock timings across operating systems and outer multiprocessing settings.
The script also verifies that LOCAL, SAFE005 and SAFE010 reproduce the original
objective values at all 384 sampled reference states.

The external strong-CHILS solver is not rerun here; its measured wall times are
read from red2pack_benchmark_output/red2pack_results.csv. The dynamic methods
are effectively single-threaded, while the external reference is allowed its
native OpenMP configuration.

Dependencies: numpy, pandas, networkx. The red2pack result file must already
exist from red2pack_benchmark.py.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Keep this run sequential. Dynamic methods are effectively single-threaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import large_scale_benchmark as bench


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "same_environment_timing_output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

BENCHMARK_REFERENCE = bench.OUTPUT_DIR / "reference_points.csv"
RED2PACK_RESULTS = (
    BASE_DIR
    / "red2pack_benchmark_output"
    / "red2pack_results.csv"
)

TASK_TIMING_CSV = OUTPUT_DIR / "task_timing.csv"
SUMMARY_CSV = OUTPUT_DIR / "timing_summary.csv"
VERIFICATION_CSV = OUTPUT_DIR / "verification.csv"

ABS_TOL = 1e-6
REL_TOL = 1e-10

METHODS = ("LOCAL", "SAFE005", "SAFE010")


# ============================================================
# HELPERS
# ============================================================

def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def all_full_tasks() -> List[bench.Task]:
    bench.RUN_MODE = "full"
    tasks, _ = bench.make_tasks()
    return tasks


def ref_rows_for_task(
    ref_df: pd.DataFrame,
    task: bench.Task,
) -> pd.DataFrame:

    q = ref_df[
        (ref_df["source"] == task.source)
        & (ref_df["graph_name"] == task.graph_name)
        & (ref_df["seed"] == task.seed)
        & (ref_df["weight_mode"] == task.weight_mode)
    ].copy()

    if task.source == "synthetic":
        q = q[q["n"].astype(int) == int(task.n)]

    if q.empty:
        raise RuntimeError(
            f"No original benchmark reference rows for {task.name}."
        )

    return q.sort_values("step").reset_index(drop=True)


def checkpoint_path(task: bench.Task) -> Path:
    return CHECKPOINT_DIR / f"{task.name}.csv"


def close_enough(a: float, b: float) -> bool:
    return abs(a - b) <= max(
        ABS_TOL,
        REL_TOL * max(1.0, abs(a), abs(b)),
    )


def reference_key(
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


# ============================================================
# ONE TASK
# ============================================================

def run_task(
    task: bench.Task,
    ref_rows: pd.DataFrame,
) -> pd.DataFrame:

    seed_mix = bench.seed_mix_for_task(task)
    rng = np.random.default_rng(seed_mix)

    G = bench.build_task_graph(
        task,
        seed_mix,
    )

    actual_n = G.number_of_nodes()
    m0 = G.number_of_edges()

    expected_n_values = sorted(
        set(int(x) for x in ref_rows["n"])
    )
    if expected_n_values != [actual_n]:
        raise AssertionError(
            f"{task.name}: n mismatch; reproduced {actual_n}, "
            f"expected {expected_n_values}."
        )

    expected_m0_values = sorted(
        set(int(x) for x in ref_rows["m0"])
    )
    if expected_m0_values != [m0]:
        raise AssertionError(
            f"{task.name}: m0 mismatch; reproduced {m0}, "
            f"expected {expected_m0_values}."
        )

    edge_pool = bench.DynamicEdgePool(G.edges())

    # Same RNG consumption order as original benchmark.
    w = bench.make_weights(
        G=G,
        mode=task.weight_mode,
        rng=rng,
    )

    S0, _ = bench.strong_static(
        G=G,
        w=w,
        seed=seed_mix + 17,
        restarts=bench.STRONG4_RESTARTS,
    )

    if not bench.is_feasible_2packing(G, S0):
        raise AssertionError(
            f"{task.name}: initial solution infeasible."
        )

    states: Dict[str, Set[int]] = {
        "LOCAL": set(S0),
        "SAFE005": set(S0),
        "SAFE010": set(S0),
    }

    refresh_intervals = {
        name: max(
            1,
            int(math.ceil(frac * m0)),
        )
        for name, frac in bench.SAFE_POLICIES.items()
    }

    total_steps = max(
        1,
        int(math.ceil(task.update_load * m0)),
    )

    expected_total = sorted(
        set(int(x) for x in ref_rows["total_steps"])
    )
    if expected_total != [total_steps]:
        raise AssertionError(
            f"{task.name}: total_steps mismatch; reproduced "
            f"{total_steps}, expected {expected_total}."
        )

    ref_by_step = {
        int(r["step"]): r
        for _, r in ref_rows.iterrows()
    }

    method_total_time = {
        name: 0.0
        for name in METHODS
    }

    verification_rows = []

    for step in range(1, total_steps + 1):

        update_type, u, v = bench.apply_random_update(
            G=G,
            edge_pool=edge_pool,
            rng=rng,
        )

        # ----------------------------------------------------
        # LOCAL
        # ----------------------------------------------------
        before = set(states["LOCAL"])

        t0 = time.perf_counter()
        new_state, _, _, _ = bench.local_dynamic_step(
            G=G,
            w=w,
            S=before,
            u=u,
            v=v,
            update_type=update_type,
        )
        elapsed = time.perf_counter() - t0

        method_total_time["LOCAL"] += elapsed
        states["LOCAL"] = new_state

        # ----------------------------------------------------
        # SAFE005 / SAFE010
        # ----------------------------------------------------
        for name in ("SAFE005", "SAFE010"):

            before = set(states[name])

            t0 = time.perf_counter()

            maintained, _, _, _ = bench.local_dynamic_step(
                G=G,
                w=w,
                S=before,
                u=u,
                v=v,
                update_type=update_type,
            )

            attempt = (
                step % refresh_intervals[name] == 0
            )

            chosen = maintained

            if attempt:
                chosen, _, _, _ = bench.safe_refresh_candidate(
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

            elapsed = time.perf_counter() - t0

            method_total_time[name] += elapsed
            states[name] = chosen

        # ----------------------------------------------------
        # Cross-platform reproduction audit at reference steps
        # ----------------------------------------------------
        if step in ref_by_step:

            ref = ref_by_step[step]

            if G.number_of_edges() != int(ref["m"]):
                raise AssertionError(
                    f"{task.name}, step={step}: edge count mismatch; "
                    f"reproduced {G.number_of_edges()}, "
                    f"expected {int(ref['m'])}."
                )

            row = {
                "source": task.source,
                "graph_name": task.graph_name,
                "n": actual_n,
                "seed": task.seed,
                "weight_mode": task.weight_mode,
                "step": step,
                "total_steps": total_steps,
                "m": G.number_of_edges(),
            }

            all_ok = True

            for name in METHODS:
                reproduced = bench.solution_weight(
                    states[name],
                    w,
                )
                expected = float(
                    ref[f"{name}_value"]
                )

                ok = close_enough(
                    reproduced,
                    expected,
                )

                row[f"{name}_reproduced_value"] = reproduced
                row[f"{name}_original_value"] = expected
                row[f"{name}_abs_diff"] = abs(
                    reproduced - expected
                )
                row[f"{name}_match"] = int(ok)

                all_ok = all_ok and ok

            row["all_method_values_match"] = int(all_ok)

            verification_rows.append(row)

            if not all_ok:
                raise AssertionError(
                    f"{task.name}, step={step}: original benchmark objective "
                    "reproduction mismatch. See verification row."
                )

        # Periodic feasibility audit outside the timing windows.
        if (
            step == 1
            or step == total_steps
            or step in ref_by_step
            or step % 250 == 0
        ):
            for name, S in states.items():
                if not bench.is_feasible_2packing(G, S):
                    raise AssertionError(
                        f"{task.name}, step={step}: "
                        f"{name} infeasible."
                    )

    # One checkpoint file contains:
    # first row = task timing, subsequent rows encoded as verification metadata
    # in separate columns. We return a normal verification dataframe and write
    # timing separately in main.
    timing_row = {
        "source": task.source,
        "graph_name": task.graph_name,
        "n": actual_n,
        "m0": m0,
        "seed": task.seed,
        "weight_mode": task.weight_mode,
        "total_steps": total_steps,
        "reference_points": len(ref_by_step),
    }

    for name in METHODS:
        total_s = method_total_time[name]
        timing_row[f"{name}_total_time_s"] = total_s
        timing_row[f"{name}_mean_time_ms"] = (
            1000.0 * total_s / total_steps
        )

    vdf = pd.DataFrame(verification_rows)

    # Store task timing fields on every verification row so one atomic
    # checkpoint is sufficient for resume/recovery.
    for k, v in timing_row.items():
        vdf[f"task__{k}"] = v

    return vdf


# ============================================================
# BUILD FINAL TABLES
# ============================================================

def build_outputs(
    checkpoint_files: List[Path],
    r2p_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    verification_frames = [
        pd.read_csv(p)
        for p in checkpoint_files
    ]

    verification = pd.concat(
        verification_frames,
        ignore_index=True,
    )

    task_cols = [
        c
        for c in verification.columns
        if c.startswith("task__")
    ]

    task_timing = (
        verification[task_cols]
        .drop_duplicates()
        .copy()
    )
    task_timing.columns = [
        c.removeprefix("task__")
        for c in task_timing.columns
    ]

    # Mean external benchmark external red2pack wall time for each matching task.
    r2p_keys = [
        "source",
        "graph_name",
        "n",
        "seed",
        "weight_mode",
    ]

    r2p_task = (
        r2p_df.groupby(
            r2p_keys,
            as_index=False,
        )
        .agg(
            R2P_reference_points=(
                "step",
                "count",
            ),
            R2P_mean_total_time_ms=(
                "R2P_total_time_s",
                lambda x: 1000.0 * x.mean(),
            ),
            R2P_median_total_time_ms=(
                "R2P_total_time_s",
                lambda x: 1000.0 * x.median(),
            ),
        )
    )

    task_timing = task_timing.merge(
        r2p_task,
        on=r2p_keys,
        how="left",
        validate="one_to_one",
    )

    if task_timing["R2P_mean_total_time_ms"].isna().any():
        raise RuntimeError(
            "Some timing audit tasks did not match external benchmark red2pack timing rows."
        )

    for name in METHODS:
        task_timing[
            f"{name}_speedup_vs_R2P_sameenv"
        ] = (
            task_timing["R2P_mean_total_time_ms"]
            / task_timing[f"{name}_mean_time_ms"]
        )

    # Source-level and synthetic-size summaries.
    rows = []

    def add_group(
        group_type: str,
        source: str,
        graph_name: str,
        n: int,
        g: pd.DataFrame,
    ) -> None:

        row = {
            "group_type": group_type,
            "source": source,
            "graph_name": graph_name,
            "n": n,
            "tasks": len(g),
            "R2P_mean_total_time_ms": float(
                g["R2P_mean_total_time_ms"].mean()
            ),
            "R2P_median_total_time_ms": float(
                g["R2P_mean_total_time_ms"].median()
            ),
        }

        for name in METHODS:
            row[f"{name}_mean_time_ms"] = float(
                g[f"{name}_mean_time_ms"].mean()
            )
            row[
                f"{name}_mean_speedup_vs_R2P_sameenv"
            ] = float(
                g[
                    f"{name}_speedup_vs_R2P_sameenv"
                ].mean()
            )
            row[
                f"{name}_median_speedup_vs_R2P_sameenv"
            ] = float(
                g[
                    f"{name}_speedup_vs_R2P_sameenv"
                ].median()
            )
            row[
                f"{name}_min_speedup_vs_R2P_sameenv"
            ] = float(
                g[
                    f"{name}_speedup_vs_R2P_sameenv"
                ].min()
            )

        rows.append(row)

    for source, g in task_timing.groupby(
        "source",
        sort=True,
    ):
        add_group(
            "source",
            source,
            "ALL",
            -1,
            g,
        )

    syn = task_timing[
        task_timing["source"] == "synthetic"
    ]
    for n, g in syn.groupby("n", sort=True):
        add_group(
            "synthetic_n",
            "synthetic",
            "ALL",
            int(n),
            g,
        )

    real = task_timing[
        task_timing["source"] == "real"
    ]
    for graph_name, g in real.groupby(
        "graph_name",
        sort=True,
    ):
        add_group(
            "real_graph",
            "real",
            graph_name,
            int(g["n"].iloc[0]),
            g,
        )

    summary = pd.DataFrame(rows)

    return (
        task_timing,
        summary,
        verification,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 94)
    print("DYNAMIC MW2PS — SAME-ENVIRONMENT TIMING + REPRODUCTION AUDIT")
    print("=" * 94)
    print(f"Python          : {sys.version.split()[0]}")
    print(f"Platform        : {platform.platform()}")
    print(f"OMP_NUM_THREADS : {os.environ.get('OMP_NUM_THREADS')}")
    print(f"original benchmark module  : {Path(bench.__file__).resolve()}")
    print(f"Output          : {OUTPUT_DIR}")
    print()

    reference_path = BENCHMARK_REFERENCE
    if not reference_path.exists():
        bundled = BASE_DIR / "results" / "large_scale" / "reference_points.csv"
        if bundled.exists():
            reference_path = bundled
        else:
            raise SystemExit(
                "Missing large-scale reference points. Run large_scale_benchmark.py "
                "or keep results/large_scale/reference_points.csv."
            )

    red2pack_results_path = RED2PACK_RESULTS
    if not red2pack_results_path.exists():
        bundled = BASE_DIR / "results" / "red2pack" / "results.csv"
        if bundled.exists():
            red2pack_results_path = bundled
        else:
            raise SystemExit(
                "Missing red2pack results. Run red2pack_benchmark.py "
                "or keep results/red2pack/results.csv."
            )

    # Original original benchmark self-test.
    bench.selftest()

    ref_df = pd.read_csv(
        reference_path
    )
    r2p_df = pd.read_csv(
        red2pack_results_path
    )

    tasks = all_full_tasks()

    real_names = sorted({
        t.graph_name
        for t in tasks
        if t.source == "real"
    })
    if real_names:
        bench.prepare_snap_datasets(
            real_names
        )

    pending = [
        task
        for task in tasks
        if not checkpoint_path(task).exists()
    ]

    print()
    print(f"Total tasks      : {len(tasks)}")
    print(f"Completed tasks  : {len(tasks) - len(pending)}")
    print(f"Pending tasks    : {len(pending)}")
    print(
        "Execution mode   : sequential (no outer multiprocessing)"
    )
    print()

    for i, task in enumerate(tasks, start=1):

        ckpt = checkpoint_path(task)

        if ckpt.exists():
            print(
                f"[{i:>2}/{len(tasks)}] "
                f"SKIP {task.name}",
                flush=True,
            )
            continue

        print(
            f"[{i:>2}/{len(tasks)}] "
            f"RUN  {task.name}",
            flush=True,
        )

        q = ref_rows_for_task(
            ref_df,
            task,
        )

        t0 = time.perf_counter()

        vdf = run_task(
            task,
            q,
        )

        wall = time.perf_counter() - t0

        atomic_write_csv(
            vdf,
            ckpt,
        )

        print(
            f"         DONE wall={wall:.2f}s, "
            f"reference checks={len(vdf)}, "
            f"all matched={bool(vdf['all_method_values_match'].all())}",
            flush=True,
        )

    checkpoint_files = sorted(
        CHECKPOINT_DIR.glob("*.csv")
    )

    if len(checkpoint_files) != len(tasks):
        raise RuntimeError(
            f"Expected {len(tasks)} task checkpoints, "
            f"found {len(checkpoint_files)}."
        )

    (
        task_timing,
        summary,
        verification,
    ) = build_outputs(
        checkpoint_files=checkpoint_files,
        r2p_df=r2p_df,
    )

    if len(verification) != 384:
        raise RuntimeError(
            f"Expected 384 reference verification rows, "
            f"found {len(verification)}."
        )

    if not (
        verification[
            "all_method_values_match"
        ] == 1
    ).all():
        raise RuntimeError(
            "At least one original benchmark objective reproduction check failed."
        )

    atomic_write_csv(
        task_timing,
        TASK_TIMING_CSV,
    )
    atomic_write_csv(
        summary,
        SUMMARY_CSV,
    )
    atomic_write_csv(
        verification,
        VERIFICATION_CSV,
    )

    print()
    print("=" * 94)
    print("SAME-ENVIRONMENT TIMING AUDIT COMPLETE")
    print("=" * 94)
    print(
        "Cross-platform reproduction: "
        "384/384 reference states matched for "
        "LOCAL, SAFE005, and SAFE010."
    )
    print(f"Task timing  : {TASK_TIMING_CSV}")
    print(f"Summary      : {SUMMARY_CSV}")
    print(f"Verification : {VERIFICATION_CSV}")
    print()

    source_rows = summary[
        summary["group_type"] == "source"
    ]

    cols = [
        "source",
        "tasks",
        "R2P_mean_total_time_ms",
        "LOCAL_mean_time_ms",
        "SAFE010_mean_time_ms",
        "SAFE005_mean_time_ms",
        "SAFE010_mean_speedup_vs_R2P_sameenv",
        "SAFE005_mean_speedup_vs_R2P_sameenv",
        "SAFE010_median_speedup_vs_R2P_sameenv",
        "SAFE005_median_speedup_vs_R2P_sameenv",
    ]

    print(source_rows[cols].to_string(index=False))
    print()
    print(f"  1) {SUMMARY_CSV.name}")
    print(f"  2) {TASK_TIMING_CSV.name}")
    print(f"  3) {VERIFICATION_CSV.name}")


if __name__ == "__main__":
    main()
