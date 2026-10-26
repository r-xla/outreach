# Generates figures/benchmark.png for the EuroBioC poster.
# Run from the quarto/ directory: Rscript benchmark.R
#
# Benchmarks the randomized SVD shown on the poster:
#   - base R: a plain-R implementation of the same algorithm
#   - anvl CPU: the jit()'d anvl version on CPU
#   - anvl GPU: the jit()'d anvl version on CUDA (if available)
#
# For anvl, JIT compile time and host<->device transfers are excluded:
# inputs are placed on the target device once, before timing.

suppressMessages(library(anvl))
#library(ggplot2)
#library(ragg)

# Sized so per-iteration work is well above system.time()'s ~ms resolution
# on every device, and so the GPU has enough flops to amortize kernel launch
# + cuSOLVER overhead and overtake the CPU.
N <- 10000L
P <- 2000L
ks <- c(250L, 500L, 1000L)
n_runs <- 10L

# --- Inputs -----------------------------------------------------------------

# Generate A as a rank-`r` signal + small Gaussian noise. This is the standard
# rsvd test matrix: a clear decaying spectrum that the truncated SVD is meant
# to capture, no degenerate collinearity, and modest norm so f32 stays
# numerically safe through QR/SVD on every backend.
set.seed(7L)
r <- 200L
U <- matrix(rnorm(N * r), N, r)
V <- matrix(rnorm(P * r), P, r)
A_base <- tcrossprod(U, V) / sqrt(r) + 0.01 * matrix(rnorm(N * P), N, P)
rm(U, V)
A_cpu <- nv_array(A_base)

# Deterministic seed across measurements -- avoids pathological RNG values
# that have tripped the CPU PJRT's SVD with gesdd info != 0.
key_cpu <- nv_rng_state(123L)

has_cuda <- pjrt::plugins_downloaded("cuda")
if (has_cuda) {
  A_gpu <- nv_array(A_base, device = "cuda")
  # Build the key directly on cuda so all inputs share a device.
  key_gpu <- jit(function() nv_rng_state(123L), device = "cuda")()
  # Make sure transfer is complete -- not part of the timed work.
  await(A_gpu$data)
  await(key_gpu$data)
}

# --- Implementations --------------------------------------------------------

# Same body as the displayed rsvd in the poster, written in plain R.
rsvd_R <- function(A, k) {
  Omega <- matrix(rnorm(ncol(A) * k), ncol(A), k)
  Q <- qr.Q(qr(A %*% Omega))
  B <- t(Q) %*% A
  s <- svd(B)
  list(d = s$d, u = Q %*% s$u, vt = t(s$v))
}

# CPU version: factorize the (k, P) wide matrix t(Q) %*% A directly
# (matches the rsvd code shown on the poster).
rsvd_cpu <- jit(
  function(A, key, k) {
    Omega <- nv_rnorm(c(ncol(A), k), key)[[2]]
    Q <- nv_qr(A %*% Omega)$Q
    B <- t(Q) %*% A
    s <- nv_svd(B)
    list(d = s$d, u = Q %*% s$u, vt = s$vt)
  },
  static = "k"
)

# CUDA version: cuSOLVER's SVD requires m >= n, so we factorize the tall
# (P, k) matrix t(A) %*% Q and swap U <-> V. If t(A) %*% Q = U' S V'^T, then
# t(Q) %*% A = V' S U'^T.
rsvd_gpu <- jit(
  function(A, key, k) {
    Omega <- nv_rnorm(c(ncol(A), k), key)[[2]]
    Q <- nv_qr(A %*% Omega)$Q
    Bt <- t(A) %*% Q
    s <- nv_svd(Bt)
    list(d = s$d, u = Q %*% t(s$vt), vt = t(s$u))
  },
  static = "k"
)

# Block until every output buffer in `out` is materialized on its device.
# Walks the result tree to its `PJRTBuffer` leaves. We dispatch on the leaf
# type rather than on the AnvilArray wrapper class because the wrapper has
# been renamed at least once -- matching on the buffer is invariant.
await_all <- function(lst) lapply(lst, await)

time_n <- function(fn, n) {
  median(replicate(n, as.numeric(system.time(fn())["elapsed"])))
}

time_anvl <- function(rsvd, A, key, k, n) {
  # Warm up: compile + first execution, not timed.
  await_all(rsvd(A, key, k))
  time_n(function() await_all(rsvd(A, key, k)), n = n)
}

---Run -
  -------------------------------------------------------------------results <- do.call(
  rbind,
  lapply(ks, function(k) {
    cat("k =", k, "...\n")

    invisible(rsvd_R(A_base, k)) # warmup
    t_R <- time_n(function() rsvd_R(A_base, k), n = n_runs)

    t_cpu <- time_anvl(rsvd_cpu, A_cpu, key_cpu, k, n_runs)

    t_gpu <- if (has_cuda) {
      time_anvl(rsvd_gpu, A_gpu, key_gpu, k, n_runs)
    } else {
      NA_real_
    }

    data.frame(
      k = k,
      impl = c("base R", "anvl CPU", "anvl GPU"),
      time_s = c(t_R, t_cpu, t_gpu)
    )
  })
)

results$impl <- factor(
  results$impl,
  levels = c("base R", "anvl CPU", "anvl GPU")
)

dir.create("figures", showWarnings = FALSE)
write.csv(results, "figures/benchmark.csv", row.names = FALSE)
cat("Wrote figures/benchmark.csv\n")


p <- ggplot(results, aes(k, time_s, color = impl, group = impl)) +
  geom_line(linewidth = 1.4, na.rm = TRUE) +
  geom_point(size = 5, na.rm = TRUE) +
  scale_color_manual(
    values = c(
      "base R" = "#888888",
      "anvl CPU" = "#1a3a5c",
      "anvl GPU" = "#c5832b"
    )
  ) +
  scale_y_log10(labels = function(x) {
    ifelse(x < 1, paste0(round(x * 1000), " ms"), paste0(round(x, 1), " s"))
  }) +
  scale_x_continuous(breaks = ks) +
  labs(
    x = "target rank k",
    y = "wall time (compilation excluded)",
    color = NULL,
    title = sprintf("Randomized SVD on a %d x %d matrix", N, P)
  ) +
  theme_minimal(base_size = 16, base_family = "Arial") +
  theme(
    plot.title = element_text(face = "bold", size = 17),
    plot.subtitle = element_text(size = 13, color = "#5a6b7d"),
    legend.position = "right",
    legend.text = element_text(size = 14),
    legend.key.width = grid::unit(1, "lines"),
    legend.margin = margin(0, 0, 0, 0),
    legend.box.margin = margin(0, 0, 0, 0),
    legend.box.spacing = grid::unit(2, "pt"),
    plot.margin = margin(2, 2, 2, 2)
  )

ggsave(
  here::here("eurobioc-poster", "figures", "benchmark.png"),
  p,
  width = 9,
  height = 5,
  dpi = 200,
  device = ragg::agg_png
)
cat("Wrote figures/benchmark.png\n")
