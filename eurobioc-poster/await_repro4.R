suppressMessages(library(anvl))

N <- 4000L
A <- nv_array(matrix(rnorm(N * N), N, N))

f_two <- jit(function(A) {
  B <- A %*% A; C <- B %*% B; D <- C %*% C
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

invisible(await(f_two(A)$D$data))  # warm up

for (i in 1:5) {
  t0 <- Sys.time()
  out <- f_two(A)
  t_dispatch <- as.numeric(Sys.time() - t0, units = "secs")

  t0 <- Sys.time()
  await_all(out)
  t_await_all <- as.numeric(Sys.time() - t0, units = "secs")

  cat(sprintf("iter %d: dispatch=%.4f  await_all=%.4f\n",
              i, t_dispatch, t_await_all))
}

cat("\n--- await ONE buffer per fresh dispatch ---\n")
for (i in 1:5) {
  t0 <- Sys.time()
  out <- f_two(A)
  t_dispatch <- as.numeric(Sys.time() - t0, units = "secs")

  t0 <- Sys.time()
  await(out$D$data)
  t_await <- as.numeric(Sys.time() - t0, units = "secs")

  cat(sprintf("iter %d: dispatch=%.4f  await(D)=%.4f\n",
              i, t_dispatch, t_await))
}
