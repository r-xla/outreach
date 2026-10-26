# Does anvl use more than one CPU core, and for what?
#
# Every timing in the deck's Stan comparison is pinned to a single thread, on
# purpose: Stan runs one chain in one thread, so a fair per-draw comparison has
# to give anvl one thread too. That is not the default, and a reader who runs
# the sampler on their own machine should know what the default does.
#
# XLA parallelises a compiled program across cores by itself, with nothing to
# configure -- but the parallelism lives in large array operations, not in
# control flow. So this measures two workloads that sit on opposite sides of
# that line:
#
#   * a dense matmul, which the Eigen thread pool splits across cores;
#   * the HMC sampler, a long sequential nv_while over small tensors, which
#     has nothing to split.
#
# PJRT_NPROC caps the CPU client's intra-op thread pool. It is read once when
# the client starts and is immutable afterwards, so each setting needs its own
# process: this script re-invokes itself as a worker for each one.
#
# Usage:  Rscript cpu-threads.R      (writes cpu-threads.csv)

worker <- Sys.getenv("CPU_THREADS_WORKER", "")
out_file <- Sys.getenv("CPU_THREADS_OUT", "cpu-threads.csv")

n_cores <- as.integer(
  if (Sys.info()[["sysname"]] == "Darwin") {
    system2("sysctl", c("-n", "hw.ncpu"), stdout = TRUE)
  } else {
    system2("nproc", stdout = TRUE)
  }
)

# --- driver: run one worker per thread setting, then merge -------------------
if (!nzchar(worker)) {
  frags <- vapply(c("1", "all"), function(setting) {
    frag <- tempfile(fileext = ".csv")
    env <- c(paste0("CPU_THREADS_WORKER=", setting),
             paste0("CPU_THREADS_FRAG=", frag))
    # PJRT_NPROC unset in the "all" worker: that IS the default being measured.
    if (setting == "1") env <- c(env, "PJRT_NPROC=1")
    status <- system2(file.path(R.home("bin"), "Rscript"), "cpu-threads.R",
                      env = env, stdout = FALSE, stderr = FALSE)
    if (status != 0L) stop("worker '", setting, "' failed")
    frag
  }, character(1))

  parts <- lapply(frags, utils::read.csv)
  one <- parts[[1L]]
  all <- parts[[2L]]
  stopifnot(identical(one$workload, all$workload))

  res <- data.frame(
    workload = one$workload,
    one_core_s = one$time_s,
    all_cores_s = all$time_s,
    speedup = one$time_s / all$time_s,
    n_cores = n_cores
  )
  utils::write.csv(res, out_file, row.names = FALSE)
  cat("Wrote", out_file, "(", n_cores, "cores )\n")
  print(res, row.names = FALSE)
  quit(save = "no")
}

# --- worker -----------------------------------------------------------------
# Matching the single-thread pinning used by the Stan benchmark: capping the
# thread pool alone still leaves Eigen's within-op parallelism in generated code.
if (identical(worker, "1")) {
  Sys.setenv(XLA_FLAGS = "--xla_cpu_multi_thread_eigen=false")
}
suppressMessages(library(anvl))

# Best of several runs, discarding the first (which pays the JIT compile).
# Takes a function, not an expression: a promise would be forced once and every
# later "iteration" would time the cached value.
warm_time <- function(f, iters = 4L) {
  t <- vapply(seq_len(iters), function(i) system.time(f())[["elapsed"]], numeric(1))
  min(t[-1L])
}

# 1. Dense matmul -- thread-pool territory. Two sizes, because whether the
# thread pool is worth using at all depends on how much work there is per op.
matmul <- jit(function(A, Om) A %*% Om)
time_matmul <- function(n, p, k) {
  withr::with_seed(1L, {
    A <- nv_array(matrix(rnorm(n * p), n, p), shape = c(n, p), dtype = "f64")
    Om <- nv_array(matrix(rnorm(p * k), p, k), shape = c(p, k), dtype = "f64")
  })
  warm_time(function() invisible(as_array(matmul(A, Om))))
}
sizes <- list(c(2000L, 1000L, 200L), c(6000L, 3000L, 600L))
t_matmul <- vapply(sizes, function(d) time_matmul(d[1L], d[2L], d[3L]), numeric(1))

# 2. The HMC sampler -- a sequential loop over (K, 2) tensors.
b <- nv_scalar(0.01, dtype = "f64")
U <- function(theta, b) {
  t1 <- theta[, 1]
  t2 <- theta[, 2]
  t1^2 / 200 + (t2 - b * t1^2 + 100 * b)^2 / 2
}
grad_U <- gradient(function(theta, b) sum(U(theta, b)), wrt = "theta")

leapfrog <- function(theta, p, b, eps, L) {
  grad <- grad_U(theta, b)$theta
  out <- nv_while(
    list(theta = theta, p = p, grad = grad, i = 0L),
    \(theta, p, grad, i) i < L,
    \(theta, p, grad, i) {
      p <- p - 0.5 * eps * grad
      theta <- theta + eps * p
      grad <- grad_U(theta, b)$theta
      p <- p - 0.5 * eps * grad
      list(theta = theta, p = p, grad = grad, i = i + 1L)
    }
  )
  list(theta = out$theta, p = out$p)
}

hmc_step <- function(theta, rng, b, eps, L) {
  d <- nv_rnorm(shape(theta), rng, dtype = "f64")
  p <- d[[2L]]
  H0 <- U(theta, b) + nv_reduce_sum(p^2, axes = 2L) / 2
  lf <- leapfrog(theta, p, b, eps, L)
  H1 <- U(lf$theta, b) + nv_reduce_sum(lf$p^2, axes = 2L) / 2
  K <- shape(theta)[1L]
  d <- nv_runif(K, d[[1L]], dtype = "f64")
  ok <- log(d[[2L]]) < H0 - H1
  ok <- nv_broadcast_to(nv_reshape(ok, c(K, 1L)), shape(theta))
  list(theta = nv_ifelse(ok, lf$theta, theta), rng = d[[1L]])
}

sample_hmc <- jit(function(theta, rng, b, eps, L, n) {
  buf <- nv_fill(0, shape = c(n, shape(theta)[1L], 2L), dtype = "f64")
  out <- nv_while(
    list(theta = theta, rng = rng, buf = buf, i = nv_scalar(0L)),
    \(theta, rng, buf, i) i < n,
    \(theta, rng, buf, i) {
      s <- hmc_step(theta, rng, b, eps, L)
      buf[i + 1L, , ] <- s$theta
      list(theta = s$theta, rng = s$rng, buf = buf, i = i + 1L)
    }
  )
  out$buf
}, static = "n")

n_chains <- 8192L
n_draws <- 200L
eps <- nv_scalar(0.2, dtype = "f64")
L <- nv_scalar(80L)
theta0 <- nv_array(matrix(0, nrow = n_chains, ncol = 2L), dtype = "f64")
run_hmc <- function() {
  invisible(as_array(sample_hmc(theta0, nv_rng_state(1L), b, eps, L, n = n_draws)))
}
t_hmc <- warm_time(run_hmc)

utils::write.csv(
  data.frame(
    workload = c(
      vapply(sizes, function(d)
        sprintf("dense matmul (%dx%d · %dx%d)", d[1L], d[2L], d[2L], d[3L]),
        character(1)),
      sprintf("HMC sampler (%s chains)", format(n_chains, big.mark = " "))
    ),
    time_s = c(t_matmul, t_hmc)
  ),
  Sys.getenv("CPU_THREADS_FRAG"), row.names = FALSE
)
