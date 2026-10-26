#!/usr/bin/env Rscript
#
# Randomized SVD benchmark for the {anvl} talk (talks/bioc-gpu-2026-user.qmd).
#
# Compares the SAME randomized-SVD algorithm implemented three ways:
#   - base R      : qr() / svd() / %*%              (always f64)
#   - anvl CPU    : jit()'d, dtype f64              (f64 to match base R)
#   - anvl CUDA   : same, on GPU (only if a CUDA plugin is available)
#
# Everything is f64. The earlier poster benchmark ran anvl in f32 against
# base R's f64, which is not a like-for-like comparison: f32 halves the memory
# traffic and, on most hardware, is substantially faster. This script fixes
# that.
#
# THREADS. We deliberately do NOT constrain them. Using every available core
# without the user configuring anything is one of anvl's genuine advantages:
# XLA parallelises the compiled program by itself, whereas R's default BLAS is
# often single-threaded. Pinning both sides to one core would measure a
# narrower question (per-core efficiency) and hide that.
#
# This makes it an end-to-end, as-installed comparison rather than an
# algorithmic one. Base R's column therefore depends heavily on which BLAS R
# was built against -- a reference-BLAS build is several times slower than
# OpenBLAS or Accelerate. Both the BLAS in use and the core count are recorded
# in the output so a row can always be interpreted.
#
# CONFIGURATION (environment variables, all optional):
#   SVD_N     rows of A                  (default 5000)
#   SVD_P     columns of A               (default 1000)
#   SVD_KS    comma-separated target ranks (default "100,200,400")
#   SVD_RUNS  timed repetitions, median reported (default 5)
#   SVD_OUT   output CSV path            (default svd-benchmark.csv)
#
# The poster used SVD_N=10000 SVD_P=2000 SVD_KS=250,500,1000 -- roughly 20x
# the work, worth using on a real machine.

suppressMessages(library(anvl))

N        <- as.integer(Sys.getenv("SVD_N", "5000"))
P        <- as.integer(Sys.getenv("SVD_P", "1000"))
ks       <- as.integer(strsplit(Sys.getenv("SVD_KS", "100,200,400"), ",")[[1L]])
n_runs   <- as.integer(Sys.getenv("SVD_RUNS", "5"))
out_path <- Sys.getenv("SVD_OUT", "svd-benchmark.csv")

# Threads actually available to this process. mcaffinity() is Linux-only
# (NULL on macOS), and SVD_THREADS lets the caller record what it pinned.
n_cores <- local({
  env <- suppressWarnings(as.integer(Sys.getenv("SVD_THREADS", "")))
  if (!is.na(env) && env > 0L) return(env)
  aff <- tryCatch(parallel::mcaffinity(), error = function(e) NULL)
  if (!is.null(aff)) length(aff) else parallel::detectCores()
})
blas    <- basename(sessionInfo()$BLAS %||% "unknown")

cat(sprintf(
  "A: %d x %d (f64)  |  ks: %s  |  runs: %d\ncores visible: %d  |  BLAS: %s\n\n",
  N, P, paste(ks, collapse = ", "), n_runs, n_cores, blas
))

# --- Input ------------------------------------------------------------------

# Rank-r signal plus small Gaussian noise: a decaying spectrum that a truncated
# SVD is actually meant to capture, without degenerate collinearity.
set.seed(7L)
r <- 200L
U <- matrix(rnorm(N * r), N, r)
V <- matrix(rnorm(P * r), P, r)
A_base <- tcrossprod(U, V) / sqrt(r) + 0.01 * matrix(rnorm(N * P), N, P)
rm(U, V)
invisible(gc())

A_cpu   <- nv_array(A_base, dtype = "f64")
key_cpu <- nv_rng_state(123L)

has_cuda <- isTRUE(try(pjrt::plugins_downloaded("cuda"), silent = TRUE))
if (has_cuda) {
  A_gpu   <- nv_array(A_base, dtype = "f64", device = "cuda")
  key_gpu <- jit(function() nv_rng_state(123L), device = "cuda")()
  await(A_gpu$data)
  await(key_gpu$data)
}

# --- Implementations --------------------------------------------------------

# Plain R, same algorithm as the anvl version below.
rsvd_R <- function(A, k) {
  Omega <- matrix(rnorm(ncol(A) * k), ncol(A), k)
  Q <- qr.Q(qr(A %*% Omega))
  B <- t(Q) %*% A
  s <- svd(B)
  list(d = s$d, u = Q %*% s$u, vt = t(s$v))
}

# The version shown on the slides. dtype = "f64" on nv_rnorm keeps the whole
# graph in double precision -- without it the draws default to f32 and the
# type promotion rules would pull the factorization down with them.
rsvd_cpu <- jit(
  function(A, key, k) {
    Omega <- nv_rnorm(c(ncol(A), k), key, dtype = "f64")[[2L]]
    Q <- nv_qr(A %*% Omega)$Q
    B <- t(Q) %*% A
    s <- nv_svd(B)
    list(d = s$d, u = Q %*% s$u, vt = s$vt)
  },
  static = "k"
)

# cuSOLVER's SVD requires m >= n, so on GPU we factorize the tall (P, k)
# matrix t(A) %*% Q and swap the roles of U and V afterwards:
# if t(A) %*% Q = U' S V'^T then t(Q) %*% A = V' S U'^T.
rsvd_gpu <- jit(
  function(A, key, k) {
    Omega <- nv_rnorm(c(ncol(A), k), key, dtype = "f64")[[2L]]
    Q <- nv_qr(A %*% Omega)$Q
    Bt <- t(A) %*% Q
    s <- nv_svd(Bt)
    list(d = s$d, u = Q %*% t(s$vt), vt = t(s$u))
  },
  static = "k"
)

# --- Timing helpers ---------------------------------------------------------

# anvl dispatches asynchronously, so the clock must not stop until every
# output buffer is actually materialized. Without this we would be timing the
# launch, not the computation.
await_all <- function(lst) lapply(lst, await)

time_n <- function(fn, n) {
  median(replicate(n, as.numeric(system.time(fn())["elapsed"])))
}

time_anvl <- function(rsvd, A, key, k, n) {
  time_n(function() await_all(rsvd(A, key, k)), n = n)
}

# Cost of the FIRST call for a given k: trace + lower + compile + one run.
# Only meaningful if nothing has been called with this k yet -- afterwards the
# executable is cached and this measures a warm call.
time_compile <- function(rsvd, A, key, k) {
  as.numeric(system.time(await_all(rsvd(A, key, k)))["elapsed"])
}

# --- Accuracy -----------------------------------------------------------------

# Numerical fidelity check. Comparing the randomized result against an exact
# svd() would measure the *algorithm's* approximation error (large whenever k
# is below the true rank), not whether anvl computes correctly. So we hand both
# implementations the SAME sketch matrix Omega, making the whole pipeline
# deterministic, and compare the singular values they produce. In f64 these
# should agree to near machine precision; in f32 they would not.
rsvd_fixed <- jit(function(A, Omega) {
  Q <- nv_qr(A %*% Omega)$Q
  nv_svd(t(Q) %*% A)$d
})

check_accuracy <- function(k) {
  set.seed(11L)
  Omega <- matrix(rnorm(P * k), P, k)
  d_r <- svd(t(qr.Q(qr(A_base %*% Omega))) %*% A_base, nu = 0L, nv = 0L)$d
  d_a <- as_array(await(rsvd_fixed(A_cpu, nv_array(Omega, dtype = "f64"))))
  max(abs(d_a - d_r) / pmax(d_r, .Machine$double.eps))
}

# --- Run --------------------------------------------------------------------

results <- do.call(rbind, lapply(ks, function(k) {
  cat("k =", k, "... ")

  invisible(rsvd_R(A_base, k))                          # warm up
  t_R <- time_n(function() rsvd_R(A_base, k), n = n_runs)

  t_cpu_cold <- time_compile(rsvd_cpu, A_cpu, key_cpu, k)   # includes compile
  t_cpu      <- time_anvl(rsvd_cpu, A_cpu, key_cpu, k, n_runs)

  t_gpu <- t_gpu_cold <- NA_real_
  if (has_cuda) {
    t_gpu_cold <- time_compile(rsvd_gpu, A_gpu, key_gpu, k)
    t_gpu      <- time_anvl(rsvd_gpu, A_gpu, key_gpu, k, n_runs)
  }

  rel_err <- check_accuracy(k)

  cat(sprintf(
    "base R %.3fs | anvl CPU %.3fs | max rel. err vs base R (same sketch) %.2e\n",
    t_R, t_cpu, rel_err
  ))

  data.frame(
    k          = k,
    impl       = c("base R", "anvl CPU", "anvl GPU"),
    time_s     = c(t_R, t_cpu, t_gpu),
    cold_s     = c(NA_real_, t_cpu_cold, t_gpu_cold),
    rel_err    = c(NA_real_, rel_err, NA_real_),
    n          = N,
    p          = P,
    dtype      = "f64",
    cores      = n_cores,
    blas       = blas,
    stringsAsFactors = FALSE
  )
}))

results <- results[!is.na(results$time_s), ]

cat("\n")
print(results[, c("k", "impl", "time_s", "cold_s", "rel_err")], row.names = FALSE)

write.csv(results, out_path, row.names = FALSE)
cat("\nwritten to ", out_path, "\n", sep = "")
