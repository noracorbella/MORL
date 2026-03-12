# Comparison: Algorithm Benchmarking

This module benchmarks the three main multi-objective RL algorithms against each other on an identical environment, measuring computational cost and policy agreement.

## Algorithms Compared

| # | Algorithm | Description |
|---|---|---|
| 1 | **CHVI** | Convex Hull Value Iteration — train once, extract N policies by specifying weights |
| 2 | **LG-VI** | Lexicographic Value Iteration — one run per priority ordering (6 runs for all orderings) |
| 3 | **LG Hull VI** | Lexicographic Hull VI — train once, extract N policies using lexicographic ordering |

The key trade-off: CHVI and LG Hull VI pay a higher upfront training cost but extraction is nearly free. LG-VI trains faster per run but must be re-run for each priority ordering.

## Files

### Benchmark Entry Points

| File | Description |
|---|---|
| `Benchmark1_CHVI.py` | Times CHVI training + 6 lexicographic policy extractions |
| `Benchmark2_LGVI.py` | Times 6 separate LG-VI runs (one per priority ordering) |
| `Benchmark3_LGVI_lexhull.py` | Times LG Hull VI training + 6 policy extractions |

### Core Source Files (self-contained copies)

| File | Source module |
|---|---|
| `ADS_Environment.py` | Stochastic environment (same as Stochastic/) |
| `CH_VI_stochastic_v2.py` | Convex Hull VI algorithm |
| `LG_VI_stoc_lexmax.py` | Lexicographic Value Iteration |
| `LG_VI_stoc_lexhull_v3.py` | Lexicographic Hull Value Iteration |
| `LG_utils.py` | Lexicographic utilities (`lex_max`, `lex_hull`, `generate_all_priority_orders`) |
| `CH_operations.py` | Convex hull math utilities |
| `ItemAndAgent.py`, `constants.py` | Environment definitions |
| `window.py` | Pygame visualization |

### Analysis

| File | Description |
|---|---|
| `policycomparison.py` | Loads pairs of policies and reports state-wise action agreement |
| `Main_lexmax_simple.py` | Simple runner for LG-VI with configurable priority |

### Results

| File | Description |
|---|---|
| `benchmark_results/benchmark1_CHVI.txt` | Timing for CHVI benchmark |
| `benchmark_results/benchmark2_LGVI_lexhull.txt` | Timing for 6× LG-VI benchmark |
| `benchmark_results/benchmark3_LGVI_lexhull.txt` | Timing for LG Hull VI benchmark |

## How to Run

```bash
cd Comparison

python Benchmark1_CHVI.py           # ~222 seconds
python Benchmark2_LGVI.py           # ~131 seconds
python Benchmark3_LGVI_lexhull.py   # ~139 seconds
```

Each benchmark script trains from scratch and saves results to `benchmark_results/`.

## Benchmark Results

```
BENCHMARK 1: Convex Hull Value Iteration + Extract 6 Lexicographic Policies
  CHVI training:           217.9 s
  Policy extraction (×6):   4.4 s
  TOTAL:                   222.3 s

BENCHMARK 2: 6× Lexicographic Value Iteration
  Priority [0,1,2]:        100.0 s
  Priority [0,2,1]:          4.8 s
  Priority [1,0,2]:          5.7 s
  Priority [1,2,0]:          7.3 s
  Priority [2,0,1]:          5.9 s
  Priority [2,1,0]:          6.8 s
  TOTAL:                   130.6 s  (avg 21.8 s/run)

BENCHMARK 3: Lexicographic Hull VI + Extract 6 Policies
  Iteration 1 (model build): 101.4 s
  Remaining iterations:       32.6 s
  Policy extraction (×6):     4.7 s
  TOTAL:                     139.2 s
```

**Key finding:** LG Hull VI (139 s) outperforms both CHVI (222 s) and the naive 6× LG-VI approach (131 s ≈ comparable), and like CHVI, only requires a single training run to cover all priority orderings.

## Generated Directories (not in repository)

```
allpolicies/   — 544 MB: all trained policies across all algorithms and weight combinations
policies/      — 69 MB:  benchmark-specific extracted policies and intermediate files
```

These are excluded by `.gitignore`. Run the benchmark scripts to regenerate them.
