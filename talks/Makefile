# Talk deck: bioc-gpu-2026-user.qmd
#
#   make          render the deck
#   make svd      re-run the standalone randomized-SVD benchmark (optional)
#   make gradient-cost  re-measure cost per gradient evaluation (needs rstan)
#   make cpu-threads    re-measure single-core vs all-core CPU timings
#   make clean    remove rendered output
#
# Rendering needs only hmc-results.csv; it never runs a benchmark, so it is
# fast and reproducible. The benchmark tables in the deck are generated from
# that CSV rather than typed into the slides, so they cannot drift.
#
# The SVD benchmark is NOT part of the deck. It is kept here because it is a
# self-contained, reproducible comparison worth re-running on real hardware
# (and on a GPU box, where svd-benchmark.R adds CUDA rows automatically).

QMD := bioc-gpu-2026-user.qmd

SVD_N    ?= 5000
SVD_P    ?= 1000
SVD_KS   ?= 100,200,400
SVD_RUNS ?= 5
export SVD_N SVD_P SVD_KS SVD_RUNS

# Core count is recorded in the benchmark output, not controlled. anvl using
# every core by default is part of what is being measured (see the header of
# svd-benchmark.R), so nothing is pinned here.
UNAME := $(shell uname -s)
NCPU  := $(shell if [ "$(UNAME)" = "Darwin" ]; then sysctl -n hw.ncpu; else nproc; fi)

.PHONY: all render svd svd-gpu gradient-cost cpu-threads clean distclean

# Default target. `all` is an alias -- the deck no longer depends on any
# benchmark run, so rendering is all there is to do.
all: render

render: bioc-gpu-2026-user.html

bioc-gpu-2026-user.html: $(QMD) hmc-results.csv gradient-cost.csv cpu-threads.csv
	quarto render $(QMD)

# Decomposes the anvl-vs-Stan gap into cost-per-gradient-evaluation and fixed
# per-draw cost, which is what the "where the speed-up comes from" slide reads.
# Needs rstan; takes a few minutes, mostly Stan's C++ compile.
gradient-cost:
	Rscript gradient-cost.R

# Measures what the CPU thread pool is and is not worth, by re-running two
# workloads under PJRT_NPROC=1 and under the default (every core).
cpu-threads:
	Rscript cpu-threads.R

svd:
	SVD_OUT=svd-benchmark.csv Rscript svd-benchmark.R

# On a CUDA machine: svd-benchmark.R detects the plugin and adds GPU rows.
svd-gpu:
	SVD_OUT=svd-benchmark-gpu.csv Rscript svd-benchmark.R

clean:
	rm -rf bioc-gpu-2026-user.html bioc-gpu-2026-user_files bioc-gpu-2026-user_cache

distclean: clean
	rm -f svd-benchmark-*.csv
