# Environment notes

## Original base experiments

- Operating system: Windows 11
- Python: 3.11
- Main packages: NumPy, pandas, NetworkX, SciPy/HiGHS
- Large-scale outer parallelism: three task workers

## External reference and timing audit

- Environment: WSL2 / Linux
- Python: 3.14.4 in the reviewed run
- CHSZLabLib: 0.5.27
- strong-CHILS external reference: up to 8 OpenMP threads
- LOCAL / SAFE005 / SAFE010 timing audit: sequential, effectively single-threaded

The timing comparison in the manuscript is a same-environment wall-clock
comparison. It is deliberately not described as an equal-core CPU-work
comparison. The external baseline receives the more favourable thread budget.
