suppressMessages(library(anvl))

N <- 4000L
A <- nv_array(matrix(rnorm(N * N), N, N))

f_two <- jit(function(A) {
  B <- A %*% A; C <- B %*% B; D <- C %*% C
  list(D = D, B = B)
})
invisible(await(f_two(A)$D$data))  # warmup

out <- f_two(A)
cat("Structure of out:\n")
cat(sprintf("  class(out)         = %s\n", paste(class(out), collapse = ",")))
cat(sprintf("  is.list(out)       = %s\n", is.list(out)))
cat(sprintf("  length(out)        = %d\n", length(out)))
cat(sprintf("  names(out)         = %s\n", paste(names(out), collapse = ",")))
cat(sprintf("  class(out$D)       = %s\n", paste(class(out$D), collapse = ",")))
cat(sprintf("  class(out$D$data)  = %s\n", paste(class(out$D$data), collapse = ",")))

await_all <- function(out, depth = 0) {
  pad <- strrep("  ", depth)
  cat(sprintf("%sawait_all: class=%s is.list=%s\n",
              pad, paste(class(out), collapse=","), is.list(out)))
  if (inherits(out, "AnvilArray")) {
    cat(sprintf("%s  -> calling await on $data\n", pad))
    t0 <- Sys.time()
    await(out$data)
    cat(sprintf("%s  -> await returned in %.4fs\n", pad,
                as.numeric(Sys.time() - t0, units = "secs")))
  } else if (is.list(out)) {
    lapply(out, await_all, depth = depth + 1)
  }
  invisible(NULL)
}

cat("\n--- Calling await_all on fresh dispatch ---\n")
out <- f_two(A)
await_all(out)
