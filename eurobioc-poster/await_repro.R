# Minimal repro: where does execute time go on the CPU XLA plugin?
#
# Hypothesis A: execute_async is synchronous internally on CPU.
#   -> rsvd() call itself takes ~N seconds, await() returns instantly.
# Hypothesis B: execute_async is truly async and the buffer event fires
#   only after compute completes.
#   -> rsvd() call returns ~instantly, await() takes ~N seconds.
# Hypothesis C: buffer event fires too early on CPU (bug).
#   -> rsvd() call returns ~instantly, await() also returns instantly,
#      but as_array() takes ~N seconds.

suppressMessages(library(anvl))

# Big enough to take real time: ~30 GFLOPs of compute per call.
N <- 4000L
A <- nv_array(matrix(rnorm(N * N), N, N))

f <- jit(function(A) {
  B <- A %*% A
  C <- B %*% B
  D <- C %*% C
  D
})

# Warm up: compile + first run.
invisible(as_array(f(A)))

run_one <- function(label) {
  cat(sprintf("\n--- %s ---\n", label))

  t0 <- Sys.time()
  out <- f(A)
  t_dispatch <- as.numeric(Sys.time() - t0, units = "secs")

  t0 <- Sys.time()
  await(out$data)
  t_await <- as.numeric(Sys.time() - t0, units = "secs")

  t0 <- Sys.time()
  invisible(as_array(out))
  t_as_array <- as.numeric(Sys.time() - t0, units = "secs")

  cat(sprintf("  dispatch (f()):  %8.4f s\n", t_dispatch))
  cat(sprintf("  await():         %8.4f s\n", t_await))
  cat(sprintf("  as_array():      %8.4f s\n", t_as_array))
  cat(sprintf("  total:           %8.4f s\n",
              t_dispatch + t_await + t_as_array))
}

for (i in 1:3) run_one(sprintf("iter %d", i))
