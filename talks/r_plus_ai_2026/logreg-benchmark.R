# How fast is the compiled logistic-regression fit of the R+AI deck?
#
# The slide "The whole fit, one program" fits a logistic regression with 500
# gradient-descent steps, compiled into one program with jit() and nv_while().
# This times that fit against the same 500 steps in a base R loop, with the
# gradient derived by hand, and measures how long compiling takes:
#
#   * base_r:       the base R loop, with whatever BLAS R is linked to;
#   * anvl:         the compiled fit;
#   * anvl_compile: tracing and compiling the fit, paid once per input shape.
#
# Everything runs in f64.
#
# Each time is the median over `reps` calls after one warm-up call. anvl runs
# asynchronously: its inputs are awaited before any clock starts, and every
# timed call awaits its result before the clock stops.
#
# Usage:  Rscript logreg-benchmark.R      (writes logreg-benchmark.csv)

library(anvl)

set.seed(1)
n <- 10000
p <- 10
X <- matrix(rnorm(n * p), n, p)
y <- matrix(rbinom(n, 1, plogis(X %*% rnorm(p, sd = 0.5))), ncol = 1)

# The code of the slides.
nll <- function(beta, X, y) {
  eta <- X %*% beta
  -nv_mean(y * eta - nv_log1p(nv_exp(eta)))
}
grad_nll <- jit(gradient(nll, wrt = "beta"))
fit <- jit(function(beta, X, y, lr, n_steps) {
  nv_while(
    init = list(beta = beta, i = 0L),
    cond = \(beta, i) i < n_steps,
    body = \(beta, i) {
      g <- grad_nll(beta, X, y)$beta
      list(beta = beta - lr * g, i = i + 1L)
    }
  )$beta
})

fit_r <- function(beta, X, y, lr, n_steps) {
  for (i in seq_len(n_steps)) {
    beta <- beta - lr * crossprod(X, plogis(X %*% beta) - y) / nrow(X)
  }
  beta
}

time_ms <- function(f, reps) {
  f()
  stats::median(vapply(seq_len(reps), function(i) {
    t0 <- Sys.time()
    f()
    as.numeric(Sys.time() - t0, units = "secs") * 1000
  }, numeric(1L)))
}

X64 <- nv_array(X, dtype = "f64")
y64 <- nv_array(y, dtype = "f64")
beta64 <- nv_array(matrix(0, p, 1L), dtype = "f64")
for (a in list(X64, y64, beta64)) await(a)

base_ms <- time_ms(function() fit_r(matrix(0, p, 1L), X, y, 1, 500L), reps = 20L)
f64_ms <- time_ms(function() await(fit(beta64, X64, y64, lr = 1, n_steps = 500L)), reps = 100L)

# A new shape is a new compile: drop k observations, so that every call below
# has a shape the cache has not seen. Compile time is that call minus a warm run.
new_shape_ms <- vapply(1:5, function(k) {
  Xk <- await(nv_array(X[-seq_len(k), ], dtype = "f64"))
  yk <- await(nv_array(y[-seq_len(k), , drop = FALSE], dtype = "f64"))
  t0 <- Sys.time()
  await(fit(beta64, Xk, yk, lr = 1, n_steps = 500L))
  as.numeric(Sys.time() - t0, units = "secs") * 1000
}, numeric(1L))
compile_ms <- stats::median(new_shape_ms) - f64_ms

blas <- sessionInfo()$BLAS
blas <- if (grepl("vecLib|Accelerate", blas)) {
  "Accelerate"
} else if (grepl("openblas", blas, ignore.case = TRUE)) {
  "OpenBLAS"
} else if (grepl("mkl", blas, ignore.case = TRUE)) {
  "MKL"
} else {
  "reference"
}

res <- data.frame(
  implementation = c("base_r", "anvl", "anvl_compile"),
  ms = c(base_ms, f64_ms, compile_ms),
  n = n,
  p = p,
  n_steps = 500L,
  blas = blas
)
print(res)
utils::write.csv(res, "logreg-benchmark.csv", row.names = FALSE)
