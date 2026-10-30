# Does R see the memory a Python object holds?
#
# Slide "Why not JAX via {reticulate}?" claims: re-assigning an 800 MB NumPy
# array to the same R variable grew memory by about 760 MB per iteration, until
# gc(). R only sees a small wrapper around the Python object, so its garbage
# collector has no reason to run, and the arrays that are no longer referenced
# from R stay alive in Python until it does.
#
# Two measurements per iteration:
#
#   * how many of the arrays are still alive in Python, via weak references,
#     which do not keep an array alive themselves;
#   * the memory of this process. On Linux that is the resident set size. On
#     macOS it is the physical footprint: RSS leaves out compressed pages, and
#     an array of ones compresses to almost nothing.
#
# Run with: Rscript reticulate.R
# (reticulate provisions numpy via uv; uses about 7 GB of memory at peak).
library(reticulate)
py_require("numpy")
cat("reticulate", format(packageVersion("reticulate")), "\n")
np <- import("numpy", convert = FALSE)
tracker <- py_run_string("
import weakref
refs = []
def track(x): refs.append(weakref.ref(x))
def alive(): return sum(r() is not None for r in refs)
", convert = FALSE)

# Memory of this process, in MB.
mem <- function() {
  if (file.exists("/proc/self/status")) {
    kb <- gsub("\\D", "", grep("^VmRSS", readLines("/proc/self/status"), value = TRUE))
    return(as.numeric(kb) / 1024)
  }
  header <- grep("Footprint:", system2("footprint", c("-p", Sys.getpid()), stdout = TRUE), value = TRUE)
  m <- regmatches(header, regexec("Footprint: ([0-9.]+) ([KMG]B)", header))[[1L]]
  as.numeric(m[[2L]]) * c(KB = 1 / 1024, MB = 1, GB = 1024)[[m[[3L]]]]
}

invisible(gc())
prev <- mem()
cat(sprintf("start: %.0f MB\n", prev))
for (i in 1:8) {
  a <- np$ones(100000000L) # 1e8 doubles: 800 MB
  tracker$track(a)
  now <- mem()
  cat(sprintf(
    "iteration %d: %.0f MB (%+.0f MB), arrays alive: %d, object.size(a): %s\n",
    i, now, now - prev, py_to_r(tracker$alive()), format(object.size(a))
  ))
  prev <- now
}
invisible(gc())
cat(sprintf("after gc(): %.0f MB, arrays alive: %d\n", mem(), py_to_r(tracker$alive())))
