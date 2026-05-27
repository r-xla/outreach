# Does the bug reproduce when (a) the jit output is a *list* (like rsvd) and
# (b) timing goes through system.time + replicate + median, like the benchmark?

suppressMessages(library(anvl))

N <- 4000L
A <- nv_array(matrix(rnorm(N * N), N, N))

# List-output, mirrors rsvd's return shape.
f_list <- jit(function(A) {
  B <- A %*% A
  C <- B %*% B
  D <- C %*% C
  list(D = D, B = B)
})

await_all <- function(out) {
  if (inherits(out, "AnvilArray")) {
    await(out$data)
  } else if (is.list(out)) {
    lapply(out, await_all)
  }
  invisible(NULL)
}

# Warm up
await_all(f_list(A))

# 1. Sys.time, single call
t0 <- Sys.time(); await_all(f_list(A)); cat(sprintf(
  "Sys.time, await_all on list:        %8.4f s\n",
  as.numeric(Sys.time() - t0, units = "secs")))

# 2. system.time, single call
t <- system.time(await_all(f_list(A)))
cat(sprintf("system.time, await_all on list:     %8.4f s (elapsed)\n",
            as.numeric(t["elapsed"])))

# 3. system.time inside replicate, like the benchmark
times <- replicate(5, as.numeric(system.time(await_all(f_list(A)))["elapsed"]))
cat(sprintf("replicate(5, system.time(...))->median: %8.4f s\n",
            median(times)))
cat(sprintf("  individual values: %s\n",
            paste(sprintf("%.4f", times), collapse = ", ")))

# 4. Same but await ONLY one of the outputs (test if multi-buffer matters)
await_d <- function(out) { await(out$D$data); invisible(NULL) }
times <- replicate(5, as.numeric(system.time(await_d(f_list(A)))["elapsed"]))
cat(sprintf("replicate(5, await only $D)->median:    %8.4f s\n",
            median(times)))

# 5. as_array(out$D) variant (host barrier)
sync <- function(out) { invisible(as_array(out$D)); invisible(NULL) }
times <- replicate(5, as.numeric(system.time(sync(f_list(A)))["elapsed"]))
cat(sprintf("replicate(5, as_array($D))->median:     %8.4f s\n",
            median(times)))
