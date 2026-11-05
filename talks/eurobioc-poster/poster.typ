// ===========================================================================
// EuroBioC poster (Typst)
//
// The randomized-SVD benchmark figure (figures/benchmark.png) is produced
// by benchmark.R. Regenerate with: Rscript benchmark.R
// ===========================================================================

#set page(paper: "a0", margin: 12mm, numbering: none)
#set text(font: "Arial", size: 32pt)

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
#let codebg    = rgb("#2b2b3a")
#let codefg    = rgb("#9be29b")
#let goodfill  = rgb("#e8f4ea")
#let goodline  = rgb("#a9ccaf")
#let goodink   = rgb("#2f6b3a")
#let warnfill  = rgb("#fbe9e7")
#let warnline  = rgb("#e0a89c")
#let warnink   = rgb("#9c3a28")

#set par(leading: 0.6em, justify: false)
#show link: set text(fill: navy)

// --- helpers ---------------------------------------------------------------
#let sec(title) = block(below: 22pt, above: 0pt, text(size: 42pt, weight: "bold", fill: navy)[#title])

#let band(body, fill: lightfill, line: lightline) = block(
  width: 100%, fill: fill, stroke: 1.5pt + line, radius: 8pt,
  inset: (x: 7pt, y: 12pt),
)[#body]

#let codebox(code) = block(
  width: 100%, fill: codebg, radius: 6pt, inset: 8pt,
  text(font: "JetBrains Mono", size: 36pt, fill: codefg, raw(code, lang: "r")),
)

#let codesm(code) = block(
  width: 100%, fill: codebg, radius: 5pt, inset: 7pt,
  text(font: "JetBrains Mono", size: 30pt, fill: codefg, raw(code, lang: "r")),
)

#let outbox(txt) = block(
  width: 100%, fill: white, stroke: 1pt + blueline, radius: 5pt, inset: 7pt,
  {
    set par(leading: 0.35em)
    text(font: "JetBrains Mono", size: 30pt, fill: ink, raw(txt))
  },
)

#let gap = v(14pt, weak: true)
#let dot = [#h(0.55em) · #h(0.55em)]

// ===========================================================================
// HEADER
// ===========================================================================
#block(width: 100%, fill: navy, radius: 8pt, inset: (x: 5pt, y: 15pt))[
  #grid(
    columns: (auto, 1fr), align: horizon, gutter: 30pt,
    image("assets/anvl-logo.svg", height: 180pt),
    [
      #set align(center)
      #text(size: 70pt, weight: "bold", fill: white, top-edge: "cap-height", bottom-edge: "baseline")[
        Accelerated array computing and code transformations for R
      ]
      #v(12pt)
      #text(size: 36pt, fill: white)[
        Sebastian Fischer#super[1,2] · Daniel Falbel · Tomasz Kalinowski#super[3] · Niko German · Martin Binder#super[1,2] · Bernd Bischl#super[1,2]
      ]
      #v(3pt)
      #text(size: 36pt, fill: rgb("#cdd9e6"))[
        #super[1]LMU Munich #h(0.8em) #super[2]Munich Center for Machine Learning (MCML) #h(0.8em) #super[3]Posit
      ]
    ],
  )
]

#gap

// ===========================================================================
// MOTIVATION + SHARED SETUP (side by side)
// ===========================================================================
#grid(
  columns: (1fr, 1fr), gutter: 20pt, align: top,
  band[
    #sec[Motivation]
    *anvl aspires to be for R what JAX is for Python*: write numerical
    programs in plain R, compiled via *OpenXLA* (the stack behind TensorFlow,
    JAX, PyTorch-TPU) to optimized machine code on CPU or GPU.

    Unlike eager-first frameworks such as *torch*, compilation lets anvl
    deliver performance with fewer hand-written CUDA kernels.
  ],
  band[
    #sec[Creating arrays]
    Create `AnvlArray`s from R objects with `nv_array()` — the `device` argument picks the hardware. Default `dtype` is `f32` and device `"cpu"`.
    #v(6pt)
    #codebox("X     <- nv_array(rnorm(N * P), shape = c(N, P), dtype = \"f32\")
beta  <- nv_array(rnorm(P), shape = c(P, 1), device = \"cpu\")
alpha <- nv_scalar(rnorm(1))
y     <- nv_array(rnorm(N), shape = c(N, 1))")
  ],
)

#gap

// ===========================================================================
// CAPABILITY ROW
// ===========================================================================
#grid(
  columns: (1fr, 1fr, 1fr), gutter: 20pt, align: top,
  band[
    #sec[① Eager mode]
    Each operation (`%*%`, `+`) is *dispatched and run immediately*.
    For debugging and interactive use.
    #v(6pt)
    #codebox("linear <- function(X, beta, alpha)
  X %*% beta + alpha
y_hat <- linear(X, beta, alpha)")
  ],
  band[
    #sec[② JIT mode]
    `jit()` *compiles the whole function as a single unit*, allowing for joint optimizations like
    kernel fusion or buffer reuse. Shapes and data types are specialized and subsequent
    calls with the same input shapes and dtypes use the cached executable.
    #v(6pt)
    #codebox("fast  <- jit(linear)
y_hat <- fast(X, beta, alpha)")
  ],
  band[
    #sec[③ Automatic differentiation]
    `gradient()` returns a new function that computes *reverse-mode
    gradients* of an R function.
    #v(6pt)
    #codebox("loss <- function(X, y, beta, alpha)
  mean((linear(X, beta, alpha) - y)^2)
grad <- jit(gradient(loss, wrt = \"beta\"))
g    <- grad(X, y, beta, alpha)")
  ],
)

#gap

// ===========================================================================
// WHY ANVL
// ===========================================================================
#band[
  #sec[Why anvl?]
  #grid(
    columns: (1fr, 1fr), gutter: 24pt, align: top,
    [
      - same code runs on CPU and GPU
      - a more consistent array API than base R
      - written primarily in R, so easy to extend and contribute to
      - around 140 common functions already covered and more coming
      - possible use-cases: gradient-based methods, Monte Carlo samplers, expensive matrix algebra, etc.
    ],
    [
        #box(width: 100%, fill: goodfill, stroke: 1.5pt + goodline,
            radius: 6pt, inset: 12pt)[
          #text(size: 32pt, weight: "bold", fill: goodink)[✓ Fast when]
          #v(3pt)
          #text(size: 32pt)[
            - compile overhead amortizes: repeated calls with the same shapes (padding can help) or one substantial call
            - the work involves dense arrays
          ]
        ]
        #v(8pt)
        #box(width: 100%, fill: warnfill, stroke: 1.5pt + warnline,
            radius: 6pt, inset: 12pt)[
          #text(size: 32pt, weight: "bold", fill: warnink)[✗ Not great (yet) when]
          #v(3pt)
          #text(size: 32pt)[
            - data-dependent shape computations are present (cannot be compiled)
            - the data is sparse
          ]
        ]
    ],
  )
]

#gap

// ===========================================================================
// HOW IT WORKS
// ===========================================================================
#band(fill: bluefill, line: blueline)[
  #grid(
    columns: (1.35fr, 1fr), align: horizon, gutter: 30pt,
    [
      #sec[How it works]
      #text(size: 32pt)[
        Three *code transformations*, shown below on `f <- function(x) x * 2`
        with the output of each step.
      ]
    ],
    align(center, text(size: 24pt)[#diagram(
      node-stroke: 1.5pt + navy,
      edge-stroke: 1.4pt + navy,
      spacing: (2.45cm, 1.78cm),
      node((0, 0), [R code], fill: white, corner-radius: 4pt, inset: 8pt),
      node((1, 0), [AnvlGraph\ (IR)], fill: bluefill, corner-radius: 4pt, inset: 8pt),
      node((2, 0), [Executable], fill: white, corner-radius: 4pt, inset: 8pt),
      edge((0, 0), (1, 0), "->", [trace]),
      edge((1, 0), (2, 0), "->", [lower]),
      edge((1, 0), (1, 0), "->", [transform], bend: 130deg),
    )]),
  )
  #v(6pt)
  #let step(num, title, sub, desc, call, out) = block(
    width: 100%, fill: white, stroke: 1.5pt + blueline, radius: 6pt, inset: 9pt,
  )[
    #text(size: 32pt, weight: "bold", fill: navy)[#num #h(6pt) #title]
    #h(1fr)
    #text(size: 24pt, fill: rgb("#5a6b7d"))[#sub]
    #v(5pt)
    #text(size: 32pt)[#desc]
    #v(7pt)
    #codesm(call)
    #v(6pt)
    #outbox(out)
  ]
  #grid(
    columns: (1fr, 1fr, 1.25fr), rows: auto, gutter: 12pt,
    step("①", "Trace", "R → AnvlGraph",
      [Run the function on abstract inputs and record primitive operations.],
      "trace_fn(f, args = list(
  x = nv_aval(\"f32\", integer())))",
      "<AnvlGraph>
  Inputs:
    %x1: f32[]
  Body:
    %1: f32[] = mul(%x1, 2:f32?)
  Outputs:
    %1: f32[]"),
    step("②", "Transform", "AnvlGraph → AnvlGraph",
      [Rewrite the graph, e.g. to compute its gradient.],
      "transform_gradient(graph, wrt = \"x\")",
      "<AnvlGraph>
  Inputs:
    %x1: f32[]
  Constants:
    %c1: f32[]
  Body:
    %1 = mul(%x1, 2:f32?)
    %2 = mul(%c1, 2:f32?)
  Outputs:
    %2: f32[]"),
    step("③", "Lower", "AnvlGraph → Executable",
      [Compile the graph into a hardware-specific executable.],
      "hlo <- stablehlo(graph)[[1]]
exe <- pjrt_compile(pjrt_program(repr(hlo)))
print(hlo)",
      "func.func @main(%0: tensor<f32>) -> tensor<f32> {
  %1 = stablehlo.constant dense<2.0e0> : tensor<f32>
  %2 = stablehlo.multiply %0, %1 : tensor<f32>
  return %2 : tensor<f32>
}"),
  )
]

#gap

// ===========================================================================
// APPLICATION — randomized SVD
// ===========================================================================
#band(fill: amberfill, line: amberline)[
  #sec[Application: randomized SVD]
  For $A in RR^(n times p)$ with $n,p$ large but only the top $k << min(n,p)$
  singular triplets needed, the randomized SVD approximates them in four steps:
  sample $Omega in RR^(p times k) tilde cal(N)(0,1)$;
  sketch $Y = A Omega in RR^(n times k)$;
  orthonormalize $Q R = Y$;
  and SVD the small matrix
  $B = Q^T A = tilde(U) Sigma V^T$.
  #v(8pt)
  #grid(
    columns: (1fr, 1fr), gutter: 30pt, align: top,
    [
      #codebox("k <- 200L
rsvd <- jit(function(A, key, k) {
  Omega <- nv_rnorm(c(ncol(A), k), key)[[2]]
  Q <- nv_qr(A %*% Omega)$Q
  B <- t(Q) %*% A
  s <- nv_svd(B)
  list(d = s$d, u = Q %*% s$u, vt = s$vt)
}, static = \"k\")")
      #v(4pt)
      #text(size: 24pt, fill: rgb("#5a6b7d"))[
        Halko et al. (2011),
        #link("https://arxiv.org/abs/0909.4061")[arxiv.org/abs/0909.4061].
        Based on Gundersen (2019),
        #link("https://gregorygundersen.com/blog/2019/01/17/randomized-svd/")[gregorygundersen.com/blog/2019/01/17/randomized-svd].
      ]
    ],
    [
      // Benchmark figure generated by the R chunk above. anvl CPU and base R
      // are real numbers from this machine; GPU bar is a TODO placeholder.
      #box(width: 100%, fill: white, stroke: 1.5pt + amberline,
           radius: 6pt, inset: 8pt)[
        #image("figures/benchmark.png", width: 90%, height: 480pt, fit: "contain")
      ]
    ],
  )
]

#gap

// ===========================================================================
// FOOTER — MaRDI + repo QR
// ===========================================================================
#block(width: 100%, fill: navy, radius: 8pt, inset: 5pt)[
  #grid(
    columns: (auto, 1fr, auto), align: horizon, gutter: 26pt,
    box(fill: white, radius: 6pt, inset: 12pt,
      image("assets/mardi-logo.png", height: 90pt)),
    text(size: 32pt, fill: rgb("#cdd9e6"))[
      Supported by *MaRDI* — the Mathematical Research Data Initiative, an NFDI
      project funded by DFG grant 460135501.
      #v(4pt)
      #text(fill: white, weight: "bold")[
        #link("https://github.com/r-xla/anvl")[github.com/r-xla/anvl]
      ]
    ],
    box(fill: white, radius: 6pt, inset: 6pt,
      qrcode("https://github.com/r-xla/anvl", options: (scale: 2.8))),
  )
]
