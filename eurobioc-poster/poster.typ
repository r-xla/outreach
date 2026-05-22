// EuroBioC poster for the anvl R package — A0 portrait, Typst.
// See ../docs/superpowers/specs/2026-05-22-eurobioc-poster-design.md

#import "@preview/tiaoma:0.3.0": qrcode
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

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
#let sec(title) = text(size: 42pt, weight: "bold", fill: navy)[#title]

#let band(body, fill: lightfill, line: lightline) = block(
  width: 100%, fill: fill, stroke: 1.5pt + line, radius: 8pt, inset: 7pt,
)[#body]

#let codebox(code) = block(
  width: 100%, fill: codebg, radius: 6pt, inset: 8pt,
  text(font: "JetBrains Mono", size: 23pt, fill: codefg, raw(code, lang: "r")),
)

#let chip(body) = box(
  fill: warnfill, stroke: 1.5pt + warnline, radius: 6pt, inset: 10pt,
  text(size: 26pt, fill: warnink)[#body],
)

// compact code box (R input) for the How-it-works steps
#let codesm(code) = block(
  width: 100%, fill: codebg, radius: 5pt, inset: 7pt,
  text(font: "JetBrains Mono", size: 16pt, fill: codefg, raw(code, lang: "r")),
)

// console-output box for the How-it-works steps
#let outbox(txt) = block(
  width: 100%, fill: white, stroke: 1pt + blueline, radius: 5pt, inset: 7pt,
  text(font: "JetBrains Mono", size: 16pt, fill: ink, raw(txt)),
)

// stack of bands with consistent vertical rhythm
#let gap = v(2pt, weak: true)

// inline bullet separator
#let dot = [#h(0.55em) · #h(0.55em)]

// ===========================================================================
// HEADER
// ===========================================================================
#block(
  width: 100%, fill: navy, radius: 8pt, inset: 7pt,
)[
  #grid(
    columns: (auto, 1fr), align: horizon, gutter: 30pt,
    image("assets/anvl-logo.svg", height: 130pt),
    [
      #set align(center)
      #text(size: 82pt, weight: "bold", fill: white)[
        Accelerated array computing and code transformations for R
      ]
      #v(4pt)
      #text(size: 36pt, fill: white)[
        Sebastian Fischer#super[1] · Daniel Falbel · Martin Binder#super[1] · Bernd Bischl#super[1]
      ]
      #v(3pt)
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
  #v(6pt)
  *anvl* brings *compiled, GPU-accelerated array computing* to the R community:
  you write array programs in plain R, which anvl compiles to optimized machine
  code via the *OpenXLA* compiler stack — the same infrastructure behind JAX,
  TensorFlow, and PyTorch/XLA. anvl aspires to be for R what JAX is for Python.
]

#gap

// ===========================================================================
// CAPABILITY ROW — eager & JIT  |  automatic differentiation
// ===========================================================================
#grid(
  columns: (1fr, 1fr), gutter: 20pt,
  band[
    #sec[① Execution: eager & JIT]
    #v(8pt)
    Running example — a *linear-model forward pass*:
    #v(6pt)
    #codebox("predict <- function(X, beta) X %*% beta")
    #v(5pt)
    *Eager mode* (default): each operation is dispatched and run immediately,
    optimized in isolation — ideal for interactive use and debugging.

    #v(6pt)
    *JIT mode*: `jit()` traces the whole function into a computational graph,
    which the XLA compiler optimizes as a single unit — fusing operations,
    eliminating redundant buffers, and specializing to the actual shapes and
    dtypes — before emitting one cached executable.
    #v(5pt)
    #codebox("y_hat <- predict(X, beta)  # eager: runs now
fast  <- jit(predict)      # trace + compile
y_hat <- fast(X, beta)     # cached executable")
  ],
  band[
    #sec[② Automatic differentiation]
    #v(8pt)
    `gradient()` returns a new function that computes *reverse-mode gradients*
    of a plain R function — no derivations by hand, and it composes with
    `jit()`. Here we fit the linear model above by minimizing its *mean
    squared error*:
    #v(5pt)
    #codebox("loss <- function(X, y, beta) mean((X %*% beta - y)^2)

grad <- gradient(loss, wrt = \"beta\")
g    <- grad(X, y, beta)   # gradient w.r.t. beta")
  ],
)

#gap

// ===========================================================================
// WHAT ANVL CAN DO
// ===========================================================================
#band[
  #sec[What anvl can do]
  #v(6pt)
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
  #v(6pt)
  #block(stroke: (top: 1.5pt + lightline), inset: (top: 10pt), width: 100%)[
    *Why anvl?* #h(0.4em) GPU acceleration without leaving R #dot
    a more intuitive and consistent array-computing API than base R #dot
    simplified CUDA install (only an NVIDIA driver needed) #dot
    JIT compilation sets it apart from *{torch}* #dot
    written primarily in R, so easy to extend and contribute to.
  ]
  #v(6pt)
  #chip[
    *⚠ Scope:* anvl currently targets *dense* arrays — sparse matrices are not
    well supported yet. Best fit: dense data, regular shapes, and repeated or
    batched computation.
  ]
]

#gap

// ===========================================================================
// HOW IT WORKS — three code transformations, demoed on a tiny function
// ===========================================================================
#band(fill: bluefill, line: blueline)[
  #grid(
    columns: (1.35fr, 1fr), align: horizon, gutter: 30pt,
    [
      #sec[How it works]
      #v(6pt)
      #text(size: 30pt)[
        anvl compiles your code through *three code transformations*. Below we
        follow them on a tiny example function — `f <- function(x, y) x * y + 1`
        — calling each transformation directly and showing its output.
      ]
    ],
    align(center, text(size: 24pt)[#diagram(
      node-stroke: 1.5pt + navy,
      edge-stroke: 1.4pt + navy,
      spacing: (2.7cm, 1.95cm),
      node((0, 0), [R code], fill: white, corner-radius: 4pt, inset: 10pt),
      node((1, 0), [AnvlGraph\ (IR)], fill: bluefill, corner-radius: 4pt, inset: 10pt),
      node((2, 0), [Executable], fill: white, corner-radius: 4pt, inset: 10pt),
      edge((0, 0), (1, 0), "->", [trace]),
      edge((1, 0), (2, 0), "->", [lower]),
      edge((1, 0), (1, 0), "->", [transform], bend: 130deg),
    )]),
  )
  #v(6pt)
  #let step(num, title, sub, call, out) = box(
    width: 100%, fill: white, stroke: 1.5pt + blueline, radius: 6pt, inset: 9pt,
  )[
    #text(size: 30pt, weight: "bold", fill: navy)[#num #h(6pt) #title]
    #h(1fr)
    #text(size: 22pt, fill: rgb("#5a6b7d"))[#sub]
    #v(7pt)
    #codesm(call)
    #v(6pt)
    #outbox(out)
  ]
  #grid(
    columns: (1fr, 1fr, 1fr), gutter: 16pt, align: top,
    step("①", "Trace", "R → AnvlGraph",
      "trace_fn(f, args = list(
  x = nv_aval(\"f32\", integer()),
  y = nv_aval(\"f32\", integer())))",
      "<AnvlGraph>
  Inputs:
    %x1: f32[]
    %x2: f32[]
  Body:
    %1: f32[] = mul(%x1, %x2)
    %2: f32[] = add(%1, 1:f32?)
  Outputs:
    %2: f32[]"),
    step("②", "Transform", "graph → graph",
      "transform_gradient(
  graph, wrt = c(\"x\", \"y\"))",
      "<AnvlGraph>
  Inputs:
    %x1: f32[]   %x2: f32[]
  Constants:
    %c1: f32[]
  Body:
    %1 = mul(%x1, %x2)
    %2 = add(%1, 1:f32?)
    %3 = mul(%c1, %x2)
    %4 = mul(%c1, %x1)
  Outputs:
    %3: f32[]   %4: f32[]"),
    step("③", "Lower", "graph → executable",
      "stablehlo(graph)[[1]]",
      "func.func @main
   (%0: tensor<f32>,
    %1: tensor<f32>) -> tensor<f32> {
 %2 = stablehlo.multiply %0, %1
        : tensor<f32>
 %3 = stablehlo.constant
        dense<1.0e0> : tensor<f32>
 %4 = stablehlo.add %2, %3
        : tensor<f32>
 return %4 : tensor<f32>
}"),
  )
]

#gap

// ===========================================================================
// APPLICATION — randomized SVD
// ===========================================================================
#band(fill: amberfill, line: amberline)[
  #sec[★ Application: randomized SVD for bioinformatics]
  #v(6pt)
  #grid(
    columns: (1.05fr, 1fr), gutter: 30pt,
    [
      Randomized SVD approximates the top-#emph[k] singular triplets of a large
      matrix by sketching it onto a low-dimensional subspace with a random
      Gaussian matrix $Omega$, then taking an exact SVD of the small projection.
      It is a matmul-heavy, dense workload — XLA's sweet spot — so `jit()`-ing
      it on a GPU scales these decompositions to larger data.
      #v(8pt)
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
        width: 100%, height: 80pt, fill: white,
        stroke: (paint: amberline, thickness: 2pt, dash: "dashed"),
        radius: 6pt, inset: 16pt,
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
  #v(6pt)
  #chip[
    *⚠ For Bioconductor users:* sparse single-cell / genomics matrices would
    need to be densified first.
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
      #v(6pt)
      *Install* from r-universe — CPU works out of the box; NVIDIA GPU support
      is added via the `cuda12.8` package.
      #v(6pt)
      *Runs on:* CPU — Linux, Windows, Apple Silicon #dot
      NVIDIA GPU — Linux x86-64.
      #v(6pt)
      *Status:* early-stage, not yet on CRAN — feedback and contributions
      welcome.
      #v(6pt)
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
  width: 100%, fill: navy, radius: 8pt, inset: 7pt,
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
