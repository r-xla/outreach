# What does fusion buy for an operation you write yourself?
#
# The R+AI deck claims that an operation written from elementary operations is
# as fast as a built-in under jit(), because XLA fuses it, while torch runs each
# elementary operation as a kernel of its own. This measures that on one CPU
# core, for the tanh approximation of GELU on 10 million f32 values:
#
#   * torch, composed:  gelu(x), eight separate operations;
#   * torch, built-in:  nnf_gelu(x, approximate = "tanh"), one hand-written kernel;
#   * anvl, composed:   jit(gelu)(x), fused by XLA.
#
# Both libraries get one thread. PJRT_NPROC caps XLA's CPU thread pool; it is
# read once, when the client starts, so it is set before anvl is loaded.
# torch_set_num_threads() caps torch's.
#
# Usage:  Rscript gelu-benchmark.R      (writes gelu-benchmark.csv)

Sys.setenv(PJRT_NPROC = "1")
library(anvl)
library(torch)
torch_set_num_threads(1L)

n <- 1e7
reps <- 20L
set.seed(1)
x_r <- rnorm(n)

gelu <- function(x) 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

x_t <- torch_tensor(x_r, dtype = torch_float32())
x_a <- nv_array(x_r, dtype = "f32")
gelu_jit <- jit(gelu)

# One warm-up call (it compiles the anvl program), then the mean of `reps`
# calls. anvl runs asynchronously, so its result is awaited.
ms_per_call <- function(f) {
  f()
  1000 * system.time(for (i in seq_len(reps)) f())[["elapsed"]] / reps
}

res <- data.frame(
  implementation = c("torch_composed", "torch_builtin", "anvl_jit"),
  ms_per_call = c(
    ms_per_call(function() gelu(x_t)),
    ms_per_call(function() nnf_gelu(x_t, approximate = "tanh")),
    ms_per_call(function() await(gelu_jit(x_a)))
  ),
  n = n,
  reps = reps
)

# The three must compute the same thing.
reference <- as.numeric(nnf_gelu(x_t, approximate = "tanh"))
stopifnot(
  max(abs(as.numeric(gelu(x_t)) - reference)) < 1e-5,
  max(abs(anvl::as_array(gelu_jit(x_a)) - reference)) < 1e-5
)

utils::write.csv(res, Sys.getenv("GELU_OUT", "gelu-benchmark.csv"), row.names = FALSE)
print(res)
