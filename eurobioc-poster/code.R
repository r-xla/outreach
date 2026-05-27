library(anvl)
library(pjrt)
library(stablehlo)

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------
N <- 1000L
P <- 100L
X     <- nv_array(rnorm(N * P), shape = c(N, P), dtype = "f32", device = "cpu")
beta  <- nv_array(rnorm(P), shape = c(P, 1))
alpha <- nv_scalar(rnorm(1))
y     <- nv_array(rnorm(N), shape = c(N, 1))

# ---------------------------------------------------------------------------
# Eager mode
# ---------------------------------------------------------------------------
linear <- function(X, beta, alpha)
  X %*% beta + alpha
y_hat <- linear(X, beta, alpha)

# ---------------------------------------------------------------------------
# JIT mode
# ---------------------------------------------------------------------------
fast  <- jit(linear)
y_hat <- fast(X, beta, alpha)

# ---------------------------------------------------------------------------
# Automatic differentiation
# ---------------------------------------------------------------------------
loss <- function(X, y, beta, alpha)
  mean((linear(X, beta, alpha) - y)^2)
grad <- jit(gradient(loss, wrt = "beta"))
g    <- grad(X, y, beta, alpha)

# ---------------------------------------------------------------------------
# How it works: Trace / Transform / Lower (on tiny example f(x) = x * 2)
# ---------------------------------------------------------------------------
f <- function(x) x * 2

# Trace
graph <- trace_fn(f, args = list(
  x = nv_aval("f32", integer())))

# Transform
graph_grad <- transform_gradient(graph, wrt = "x")

# Lower
hlo <- stablehlo(graph)[[1]]
exe <- pjrt_compile(pjrt_program(repr(hlo)))

# ---------------------------------------------------------------------------
# Application: randomized SVD
# ---------------------------------------------------------------------------
A   <- nv_array(rnorm(N * P), shape = c(N, P), dtype = "f32", device = "cpu")
key <- nv_rng_state(123L)

k <- 200L
rsvd <- jit(function(A, key, k) {
  Omega <- nv_rnorm(c(ncol(A), k), key)[[2]]
  Q <- nv_qr(A %*% Omega)$Q
  B <- t(Q) %*% A
  s <- nv_svd(B)
  list(d = s$d, u = Q %*% s$u, vt = s$vt)
}, static = "k")

res <- rsvd(A, key, k)
