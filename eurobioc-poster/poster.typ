// EuroBioC poster for the anvl R package — A0 portrait, Typst.
// See ../docs/superpowers/specs/2026-05-22-eurobioc-poster-design.md

#import "@preview/tiaoma:0.3.0": qrcode

// --- palette ---------------------------------------------------------------
#let navy      = rgb("#1a3a5c")
#let ink       = rgb("#222222")
#let lightfill = rgb("#f0f3f6")
#let lightline = rgb("#d5dde5")
#let bluefill  = rgb("#eaf0f6")
#let blueline  = rgb("#b9cbdc")
#let amberfill = rgb("#fdf2e0")
#let amberline = rgb("#e8c98a")
#let warnfill  = rgb("#fbe9e7")
#let warnline  = rgb("#e0a89c")
#let warnink   = rgb("#9c3a28")
#let codebg    = rgb("#2b2b3a")
#let codefg    = rgb("#9be29b")

// --- page ------------------------------------------------------------------
#set page(width: 841mm, height: 1189mm, margin: 12mm)
#set text(font: "Arial", size: 32pt, fill: ink, lang: "en")
#set par(leading: 0.6em, justify: false)
#show link: set text(fill: navy)

// --- helpers ---------------------------------------------------------------
#let sec(title) = text(size: 46pt, weight: "bold", fill: navy)[#title]

#let band(body, fill: lightfill, line: lightline) = block(
  width: 100%, fill: fill, stroke: 1.5pt + line, radius: 8pt, inset: 22pt,
)[#body]

#let codebox(code) = block(
  width: 100%, fill: codebg, radius: 6pt, inset: 16pt,
  text(font: "JetBrains Mono", size: 27pt, fill: codefg, raw(code, lang: "r")),
)

#let chip(body) = box(
  fill: warnfill, stroke: 1.5pt + warnline, radius: 6pt, inset: 14pt,
  text(size: 28pt, fill: warnink)[#body],
)

// stack of bands with consistent vertical rhythm
#let gap = v(13pt, weak: true)

// inline bullet separator
#let dot = [#h(0.55em) · #h(0.55em)]

// ===========================================================================
// HEADER
// ===========================================================================
#block(
  width: 100%, fill: navy, radius: 8pt, inset: 24pt,
)[
  #grid(
    columns: (auto, 1fr), align: horizon, gutter: 30pt,
    image("assets/anvl-logo.svg", height: 170pt),
    [
      #set align(center)
      #text(size: 82pt, weight: "bold", fill: white)[
        Accelerated array computing and code transformations for R
      ]
      #v(12pt)
      #text(size: 36pt, fill: white)[
        Sebastian Fischer#super[1] · Daniel Falbel · Martin Binder#super[1] · Bernd Bischl#super[1]
      ]
      #v(6pt)
      #text(size: 32pt, fill: rgb("#cdd9e6"))[
        #super[1]LMU Munich, Munich Center for Machine Learning (MCML)
      ]
    ],
  )
]

#gap

// ===========================================================================
// MOTIVATION
// ===========================================================================
#band[
  #sec[Motivation]
  #v(10pt)
  *anvl* brings *compiled, GPU-accelerated array computing* to the R community.
  You write array programs in plain R, which anvl compiles under the hood to
  optimized machine code via the *OpenXLA* compiler stack — the same
  infrastructure behind JAX, TensorFlow, and PyTorch/XLA. In short, anvl
  aspires to be for R what JAX is for Python.
]

#gap

// ===========================================================================
// CAPABILITY ROW — eager & JIT  |  automatic differentiation
// ===========================================================================
#grid(
  columns: (1fr, 1fr), gutter: 20pt,
  band[
    #sec[① Execution: eager & JIT]
    #v(10pt)
    *Eager mode* (default): each operation runs immediately — ideal for
    interactive exploration and debugging.

    #v(6pt)
    *JIT mode*: `jit()` compiles a whole function into one fused executable.
    XLA optimizes across operations, which is where the speedups come from.
    #v(12pt)
    #codebox("y    <- predict(X, beta)  # eager: runs now
fast <- jit(predict)       # compile
y    <- fast(X, beta)      # fast")
  ],
  band[
    #sec[② Automatic differentiation]
    #v(10pt)
    `gradient()` returns a new function that computes *reverse-mode gradients*
    of a plain R function — no derivations by hand. It composes with `jit()`,
    so a loss, its gradient, and a training loop are all plain R.
    #v(12pt)
    #codebox("grad <- gradient(loss, wrt = \"beta\")
g    <- grad(X, y, beta)")
  ],
)

#gap

// ===========================================================================
// WHAT ANVL CAN DO
// ===========================================================================
#band[
  #sec[What anvl can do]
  #v(10pt)
  #grid(
    columns: (1fr, 1fr), gutter: 30pt,
    [
      *≈ 140 array operations.* Construction, arithmetic, linear algebra
      (`nv_matmul`, `nv_qr`, `nv_svd`, `nv_chol`, `nv_solve`), reductions and
      statistics, random-number generation, and JIT-able control flow
      (`nv_while`).
    ],
    [
      *Build, in plain R.* Neural networks, matrix factorizations, custom
      optimizers, gradient-based estimators, and Monte Carlo samplers — all
      expressed directly in R and run on CPU or GPU.
    ],
  )
  #v(16pt)
  #block(stroke: (top: 1.5pt + lightline), inset: (top: 16pt), width: 100%)[
    *Why anvl?* #h(0.4em) GPU acceleration without leaving R #dot
    simplified CUDA install (only an NVIDIA driver needed) #dot
    JIT compilation sets it apart from *{torch}* #dot
    written primarily in R, so easy to extend and contribute to.
  ]
  #v(16pt)
  #chip[
    *⚠ Scope:* anvl currently targets *dense* arrays — sparse matrices are not
    well supported yet. Best fit: dense data, regular shapes, and repeated or
    batched computation.
  ]
]

#gap

// ===========================================================================
// HOW IT WORKS — slim band
// ===========================================================================
#band(fill: bluefill, line: blueline)[
  #grid(
    columns: (auto, 1fr), align: horizon, gutter: 26pt,
    text(size: 36pt, weight: "bold", fill: navy)[How it works],
    [
      #set align(center)
      #box(fill: white, stroke: 1.5pt + blueline, radius: 6pt, inset: 14pt,
        image("assets/three-transformations.svg", height: 150pt))
      #v(8pt)
      #text(size: 28pt)[
        *Trace* R code → AnvlGraph (IR) #dot
        *Transform* graph → graph (e.g. `gradient()`) #dot
        *Lower* graph → device executable (via OpenXLA)
      ]
    ],
  )
]

#gap

// ===========================================================================
// APPLICATION — randomized SVD
// ===========================================================================
#band(fill: amberfill, line: amberline)[
  #sec[★ Application: randomized SVD for bioinformatics]
  #v(10pt)
  #grid(
    columns: (1.05fr, 1fr), gutter: 30pt,
    [
      Randomized SVD approximates the top-#emph[k] singular triplets of a large
      matrix by first sketching it onto a low-dimensional subspace with a random
      Gaussian matrix $Omega$, then doing an exact SVD of the small projection.
      It is a matmul-heavy, dense workload — exactly XLA's sweet spot — so
      `jit()`-compiling it on a GPU scales these decompositions to larger data.
      #v(12pt)
      #codebox("rsvd <- jit(function(A, Omega) {
  Q <- nv_qr(nv_matmul(A, Omega))$Q   # sketch + orthonormalize
  B <- nv_matmul(nv_transpose(Q), A)  # project (small)
  s <- nv_svd(B)                      # exact SVD of B
  list(d = s$d, u = nv_matmul(Q, s$u), vt = s$vt)
})")
    ],
    [
      #set align(center + horizon)
      #block(
        width: 100%, height: 330pt, fill: white,
        stroke: (paint: amberline, thickness: 2pt, dash: "dashed"),
        radius: 6pt, inset: 24pt,
      )[
        #text(size: 32pt, weight: "bold", fill: navy)[Benchmark figure]
        #v(8pt)
        #text(size: 28pt, fill: rgb("#888888"))[
          Placeholder — runtime of randomized SVD\
          (base R vs. anvl CPU vs. anvl GPU)\
          to be added once data is available.
        ]
      ]
    ],
  )
  #v(16pt)
  #chip[
    *⚠ For Bioconductor users:* sparse single-cell / genomics matrices would
    need densifying first — native sparse support is future work.
  ]
]

#gap

// ===========================================================================
// GETTING STARTED
// ===========================================================================
#band[
  #grid(
    columns: (1fr, auto), align: horizon, gutter: 30pt,
    [
      #sec[Getting started]
      #v(10pt)
      *Install* from r-universe — CPU works out of the box; NVIDIA GPU support
      is added via the `cuda12.8` package.
      #v(6pt)
      *Runs on:* CPU — Linux, Windows, Apple Silicon #dot
      NVIDIA GPU — Linux x86-64.
      #v(6pt)
      *Status:* early-stage, not yet on CRAN — feedback and contributions
      welcome.
      #v(10pt)
      #text(weight: "bold")[
        #link("https://github.com/r-xla/anvl")[github.com/r-xla/anvl]
        #h(28pt)
        #link("https://r-xla.github.io/anvl")[r-xla.github.io/anvl]
      ]
    ],
    box(fill: white, stroke: 1.5pt + lightline, radius: 6pt, inset: 14pt,
      qrcode("https://github.com/r-xla/anvl", options: (scale: 6.0))),
  )
]

#gap

// ===========================================================================
// FOOTER
// ===========================================================================
#block(
  width: 100%, fill: navy, radius: 8pt, inset: 22pt,
)[
  #grid(
    columns: (auto, 1fr), align: horizon, gutter: 26pt,
    box(fill: white, radius: 6pt, inset: 12pt,
      image("assets/mardi-logo.png", height: 90pt)),
    text(size: 28pt, fill: rgb("#cdd9e6"))[
      Supported by *MaRDI* — the Mathematical Research Data Initiative, an NFDI
      project funded by DFG grant 460135501.
    ],
  )
]
