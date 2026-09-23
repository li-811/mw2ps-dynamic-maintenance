# Dynamic Maximum-Weight 2-Packing under Edge Updates

Anonymous reproducibility package for a double-blind manuscript on maintaining
high-quality maximum-weight 2-packings under dynamic edge insertions and
deletions.

## What is included

The repository contains only the code needed to reproduce reported experiments
and manuscript figures. Exploratory, superseded, broken, debugging and
presentation-only development files have been removed.

- `exact_calibration.py` — exact small-instance static and dynamic calibration.
- `safe_refresh_calibration.py` — exact calibration of safe refresh periods.
- `large_scale_benchmark.py` — 96-task synthetic/SNAP dynamic benchmark.
- `red2pack_pilot.py` — 30-state external-solver budget/configuration pilot.
- `red2pack_benchmark.py` — 384-state full strong-CHILS external reference.
- `same_environment_timing.py` — same-WSL timing and 384-state reproduction audit.
- `make_paper_outputs.py` — regenerates all seven manuscript figures from bundled result tables.
- `results/` — processed outputs used in the manuscript.
- `figures/` — generated colour-online figures with redundant hatches/markers/line styles for greyscale print reproduction.

The computational kernels have not been redesigned for this release. Cleanup is
conservative: filenames, documentation, output paths and comments were
normalised, while the validated algorithms, seeds, parameters and experiment
logic were preserved.

## Reproduction order

### Base experiments (Python 3.11; Windows or Linux)

```bash
python exact_calibration.py
python safe_refresh_calibration.py
python large_scale_benchmark.py
```

Each script has `RUN_MODE = "quick"` / `"full"` near the top. The published
results use `"full"`. The large-scale script downloads the four SNAP edge lists
when they are not already present.

### External red2pack experiments (WSL2/Linux)

Install the packages in `requirements-red2pack.txt`, then run:

```bash
python red2pack_pilot.py
python red2pack_benchmark.py
python same_environment_timing.py
```

The published external configuration is strong-CHILS with strong reductions,
a five-second limit and up to eight OpenMP threads. The dynamic methods in the
timing audit are effectively single-threaded. Reported speedups are therefore
wall-clock latency ratios under the evaluated implementations, not equal-core
CPU-work ratios.

### Figures

The bundled processed CSV files are sufficient to regenerate the manuscript
figures without rerunning the long experiments. The plotting script uses a
colour-blind-friendly palette for online reading while retaining redundant
hatches, markers and line styles so the figures remain readable in greyscale:

```bash
python make_paper_outputs.py
```

## Data

Synthetic graphs are generated deterministically from fixed seeds. Public real
network topologies are downloaded from the Stanford Large Network Dataset
Collection (SNAP) by `large_scale_benchmark.py`. Directed Gnutella inputs are
converted to simple undirected graphs; self-loops and duplicate edges are
removed and vertices are compactly relabelled.

## External solver

The weighted static reference uses CHSZLabLib / red2pack. The reviewed run used
`chszlablib==0.5.27`. Floating degree-correlated node weights are encoded as
`round(10000*w)` for red2pack and the returned vertex set is evaluated again
with the original floating-point objective.

## Reproducibility checks

The same-environment audit reproduces all 384 sampled LOCAL/SAFE005/SAFE010
objective values from the original benchmark. The bundled
`results/timing/verification.csv` records these comparisons.

SHA-256 hashes for the processed result tables are provided in `SHA256SUMS`.

## Double-blind review

This package intentionally contains no author names, affiliations, email
addresses, local absolute paths, repository history or machine usernames.
Before uploading it to a review repository, do not include an existing `.git`
directory. Create the review repository from these clean files so personal Git
metadata cannot leak through commit history.

## Pilot-result note

The state-level pilot CSV was not available when this package was assembled.
The aggregate pilot summary printed by the completed run is included under
`results/pilot/summary.csv`, and the full 30-state pilot is exactly reproducible
with `red2pack_pilot.py`. For the strongest archival package, add the original
`pilot_results.csv` before making the anonymous repository public.
