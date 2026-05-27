suppressMessages(library(anvl))

N <- 4000L
A <- nv_array(matrix(rnorm(N * N), N, N))

f_two <- jit(function(A) {
  B <- A %*% A
  C <- B %*% B
  D <- C %*% C
  list(D = D, B = B)
})
invisible(await(f_two(A)$D$data))  # warmup

# Time the two awaits SEPARATELY in one call
out <- f_two(A)
t1 <- Sys.time()
await(out$D$data)
t_d <- as.numeric(Sys.time() - t1, units = "secs")
t1 <- Sys.time()
await(out$B$data)
t_b <- as.numeric(Sys.time() - t1, units = "secs")
cat(sprintf("await(D): %.4f s\n", t_d))
cat(sprintf("await(B): %.4f s\n", t_b))

# Reverse order
out <- f_two(A)
t1 <- Sys.time()
await(out$B$data)
t_b <- as.numeric(Sys.time() - t1, units = "secs")
t1 <- Sys.time()
await(out$D$data)
t_d <- as.numeric(Sys.time() - t1, units = "secs")
cat(sprintf("await(B) first:  %.4f s\n", t_b))
cat(sprintf("then await(D):   %.4f s\n", t_d))

# is_ready of each immediately after dispatch
out <- f_two(A)
cat(sprintf("immediately is_ready(D): %s\n", is_ready(out$D$data)))
cat(sprintf("immediately is_ready(B): %s\n", is_ready(out$B$data)))

# What about reading B (the small/early output) first via as_array
out <- f_two(A)
t1 <- Sys.time(); invisible(as_array(out$B)); t_b <- as.numeric(Sys.time() - t1, units = "secs")
t1 <- Sys.time(); invisible(as_array(out$D)); t_d <- as.numeric(Sys.time() - t1, units = "secs")
cat(sprintf("as_array(B) first: %.4f s\n", t_b))
cat(sprintf("then as_array(D):  %.4f s\n", t_d))
