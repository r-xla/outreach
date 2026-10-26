# Where the anvl-vs-Stan HMC gap actually comes from.
#
# The headline table (hmc-results.csv) compares whole sampling runs, which
# bundles together start-up, per-draw bookkeeping and the gradient evaluations
# themselves. This script separates them, because the interesting claim is
# about the last one.
#
# Method: hold everything fixed and vary only the number of leapfrog steps per
# draw, L. Each leapfrog step costs exactly one gradient evaluation, so
#
#     time(L) = fixed_per_draw * n + ns_per_gradient * n * L
#
# and a regression of wall time on the realised L gives the marginal cost of
# one gradient evaluation as the slope, free of every fixed cost.
#
# Stan is driven through rstan here rather than CmdStan: we want the in-process
# cost with no subprocess launch and no serialisation of draws to a text file,
# so that whatever gap remains is arithmetic and automatic differentiation and
# nothing else. Stan's gradient is the same generated C++ either way.
#
# IMPORTANT: rstan does not honour control$stepsize for a non-adapting run --
# it runs Stan's initial step-size search anyway -- so the requested L is not
# the L that ran. The realised value is read back from int_time__ / stepsize__
# and the regression uses that. Assuming it would misattribute the entire
# result.
#
# Usage:  Rscript gradient-cost.R     (writes gradient-cost.csv)

Sys.setenv(
  PJRT_NPROC = "1",
  XLA_FLAGS = "--xla_cpu_multi_thread_eigen=false",
  STAN_NUM_THREADS = "1"
)

out_file <- Sys.getenv("GRADCOST_OUT", "gradient-cost.csv")

b_param <- 0.01
eps_val <- 0.2
n_warmup <- 500L

# Best of several runs, discarding the first: the first pays one-off costs
# (JIT compile, page cache) that are not what is being measured here.
# Takes a function, not an expression -- a promise would be forced once and
# every later "iteration" would time the cached value instead of the work.
warm_time <- function(f, iters = 4L) {
  t <- vapply(seq_len(iters), function(i) system.time(f())[["elapsed"]], numeric(1))
  min(t[-1L])
}

# Slope and intercept of time ~ leapfrog steps, converted to per-draw units.
decompose <- function(leapfrog, time_s, n_draws) {
  m <- stats::lm(time_s ~ leapfrog)
  list(
    ns_per_gradient = coef(m)[[2L]] / n_draws * 1e9,
    us_fixed_per_draw = coef(m)[[1L]] / n_draws * 1e6,
    r_squared = summary(m)$r.squared
  )
}

# --- anvl -------------------------------------------------------------------
library(anvl)

b_t <- nv_scalar(b_param, dtype = "f64")

U <- function(theta, b) {
  theta1 <- theta[, 1]
  theta2 <- theta[, 2]
  theta1^2 / 200 + (theta2 - b * theta1^2 + 100 * b)^2 / 2
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

hmc_step <- function(theta, rng_state, b, eps, L) {
  d <- nv_rnorm(shape(theta), rng_state, dtype = "f64")
  p <- d[[2L]]
  H0 <- U(theta, b) + nv_reduce_sum(p^2, axes = 2L) / 2
  lf <- leapfrog(theta, p, b, eps, L)
  H1 <- U(lf$theta, b) + nv_reduce_sum(lf$p^2, axes = 2L) / 2
  K <- shape(theta)[1L]
  d <- nv_runif(K, d[[1L]], dtype = "f64")
  ok <- log(d[[2L]]) < H0 - H1
  ok <- nv_broadcast_to(nv_reshape(ok, c(K, 1L)), shape(theta))
  list(theta = nv_ifelse(ok, lf$theta, theta), rng_state = d[[1L]])
}

sample_hmc <- jit(function(theta, rng_state, b, eps, L, n) {
  buf <- nv_fill(0, shape = c(n, shape(theta)[1L], 2L), dtype = "f64")
  out <- nv_while(
    list(theta = theta, rng_state = rng_state, buf = buf, i = nv_scalar(0L)),
    \(theta, rng_state, buf, i) i < n,
    \(theta, rng_state, buf, i) {
      s <- hmc_step(theta, rng_state, b, eps, L)
      buf[i + 1L, , ] <- s$theta
      list(theta = s$theta, rng_state = s$rng_state, buf = buf, i = i + 1L)
    }
  )
  out$buf
}, static = "n")

n_anvl <- 10000L
run_anvl <- function(L_val) {
  theta <- nv_array(matrix(0, nrow = 1L, ncol = 2L), dtype = "f64")
  eps <- nv_scalar(eps_val, dtype = "f64")
  L <- nv_scalar(as.integer(L_val))
  invisible(as_array(sample_hmc(theta, nv_rng_state(42L), b_t, eps, L,
                                n = n_anvl)))
}

Ls <- c(10L, 20L, 40L, 80L, 160L)
anvl_times <- vapply(Ls, function(L) {
  invisible(run_anvl(L)) # compile for this L before timing it
  warm_time(function() run_anvl(L))
}, numeric(1))
anvl_fit <- decompose(Ls, anvl_times, n_anvl)

cat("anvl (n =", n_anvl, "draws)\n")
print(data.frame(leapfrog = Ls, time_s = anvl_times), row.names = FALSE)

# --- Stan -------------------------------------------------------------------
suppressMessages(library(rstan))
rstan::rstan_options(auto_write = TRUE)

stan_code <- "
data { real b; }
parameters { vector[2] theta; }
model {
  target += -square(theta[1]) / 200
            - square(theta[2] - b * square(theta[1]) + 100 * b) / 2;
}
"
stan_file <- file.path(tempdir(), "banana.stan")
writeLines(stan_code, stan_file)
stan_model_banana <- rstan::stan_model(file = stan_file)

n_stan <- 20000L
run_stan <- function(int_time) {
  suppressWarnings(rstan::sampling(
    stan_model_banana,
    data = list(b = b_param),
    chains = 1L, iter = n_warmup + n_stan, warmup = n_warmup, cores = 1L,
    algorithm = "HMC",
    control = list(adapt_engaged = FALSE, stepsize = eps_val,
                   int_time = int_time, metric = "unit_e"),
    refresh = 0, seed = 42L, save_warmup = FALSE
  ))
}

# The L that actually ran, not the one requested.
realised_leapfrog <- function(fit) {
  sp <- rstan::get_sampler_params(fit, inc_warmup = FALSE)[[1L]]
  mean(sp[, "int_time__"] / sp[, "stepsize__"])
}

invisible(run_stan(16)) # prime

int_times <- c(4, 8, 16, 32, 64)
stan_times <- vapply(int_times, function(it) {
  warm_time(function() invisible(run_stan(it)), iters = 3L)
}, numeric(1))
stan_leapfrog <- vapply(int_times, function(it) realised_leapfrog(run_stan(it)),
                        numeric(1))
stan_fit <- decompose(stan_leapfrog, stan_times, n_stan + n_warmup)

cat("\nStan (n =", n_stan, "draws)\n")
print(data.frame(int_time = int_times, leapfrog_realised = stan_leapfrog,
                 time_s = stan_times), row.names = FALSE)

# --- write ------------------------------------------------------------------
res <- data.frame(
  system = c("anvl", "stan"),
  ns_per_gradient = c(anvl_fit$ns_per_gradient, stan_fit$ns_per_gradient),
  us_fixed_per_draw = c(anvl_fit$us_fixed_per_draw, stan_fit$us_fixed_per_draw),
  r_squared = c(anvl_fit$r_squared, stan_fit$r_squared),
  n_draws = c(n_anvl, n_stan)
)
utils::write.csv(res, out_file, row.names = FALSE)
cat("\nWrote", out_file, "\n")
print(res, row.names = FALSE)
