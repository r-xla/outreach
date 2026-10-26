# HMC benchmark results from the anvl paper

Raw results of the Hamiltonian Monte Carlo benchmark in the anvl paper
(Bayesian logistic regression; anvl vs. CmdStan, TMB and torch, on CPU and on an
NVIDIA A100). `r-ai-2026.qmd` reads its benchmark tables from these files.

Copied unchanged from `paper/benchmarks/results/` on the anvl branch
`paper/finish-r-journal` (commit `49f7c8d8`). The scripts that produced them,
the Docker image and the LRZ job files are in `paper/benchmarks/` on that
branch. Each configuration was run three times after a first call that
includes compilation; the deck reports the median.
