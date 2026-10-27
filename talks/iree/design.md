# Slide Design: anvl at the IREE Community Meeting

Design document for the talk *"anvl – Accelerated array computing and code
transformations for R"* to the IREE developer community. The slides themselves
are not written yet. This document fixes what goes on them, in what order, and
why.

## 1. Audience and Goals

**Audience.** Compiler engineers working on IREE and MLIR. They know
StableHLO, linalg, dispatch formation, the HAL and PJRT. Most of them do not
know R or statistics, and do not need to be sold on JIT compilation.

**What they should leave with:**

1. anvl is a real frontend: an R tracer with its own IR, graph-rewrite
   transformations (reverse-mode AD), and lowering to StableHLO. Its
   architecture is close to JAX's.
2. Statistics needs dynamic shapes in a way that ML training does not, it
   mostly runs on CPU, and it needs precision: `f64` is the default, not an
   option. Recompiling for every `(n, p)` is the main thing blocking anvl's
   use for statistics today.
3. A good IREE PJRT plugin would serve more than anvl. Every frontend that
   targets PJRT (JAX, zml, GoMLX, gx, Reactant.jl, anvl) would get IREE's
   targets without changing anything on its side.
4. There are two open design questions the IREE community has to answer with
   us: in a dynamically shaped program, **who checks shape constraints such as
   `ncol(A) == nrow(B)`, and in which IR are they written down?** And **how do
   library kernels (LAPACK, cuSOLVER) reach a program that IREE compiles?**
5. This has been tried before. JAX ran on IREE from 2021 to 2023, including a
   dynamic-shapes prototype, and several of the gaps left open then are
   exactly the ones anvl hits now. Most of today's active IREE contributors
   joined after 2023, so this history is new to them.

**What I want back from them** (the talk should end with these questions):

- Who owns the IREE PJRT plugin? It is in-tree and CI-tested, but stuck at
  PJRT API 0.68 while XLA is at 0.117.
- Which input IR should a frontend emit for symbolic shapes?
- How should custom calls (LAPACK / cuSOLVER kernels) be brought to IREE?
- How good are IREE's CPU codegen and compile time for small `f64` programs,
  and should `f64` really be demoted to `f32` by default?

**Tone.** Technical and concrete. Show real IR dumps, not marketing diagrams.
The front-end demo exists to make the rest believable, and should be short.

## 2. Format and Constraints

- **Length:** assumed 25 min talk + discussion. Section 4 has a time budget;
  section 9 lists what to cut for a 15-minute slot.
- **Tooling:** Quarto `clean-revealjs`, as in `../bioc-gpu-2026-user.qmd`, with
  R chunks evaluated at render time so every trace and IR dump on a slide is
  real output. The extension lives in `../_extensions/`, so either put the
  `.qmd` in `talks/` or make `_extensions` reachable from `talks/iree/`.
- **Code style on slides:** R code short (≤ 10 lines). IR dumps may be longer
  but are cropped to the interesting part. Use `{.smaller}` sparingly.
- **Reuse:** `openxla-stack.tex` (stack diagram) and
  `three-transformations.tex` (trace → transform → lower) from the earlier
  talk, restyled where needed.

## 3. Running Example

One example runs through the whole talk, so that the demo, the IR and the
shape question all refer to the same code: **logistic regression**.

```r
nll <- function(beta, X, y) {
  eta <- X %*% beta
  sum(nv_log1p(exp(eta)) - y * eta)
}
grad_nll <- jit(gradient(nll, wrt = "beta"))
```

Why this one:

- A statistician recognises it at once, and a compiler engineer can read it
  without knowing R.
- It contains a matmul `X[n, p] %*% beta[p]`, which is the shape constraint
  of part 7 (`ncol(X) == length(beta)`, `nrow(X) == length(y)`).
- Its gradient is a short, readable graph rewrite.
- `n` and `p` are exactly the axes that vary across fits in practice
  (cross-validation folds, model selection, per-group fits).

The demo slide extends it into a full fit (a gradient-descent or Newton loop
with `nv_while()`) to show control flow. If `nv_while()` makes the IR dump too
long, show the loop body only.

## 4. Narrative Arc and Time Budget

| # | Part | Slides | Time |
|---|------|--------|------|
| 0 | Title | 1 | 0:30 |
| 1 | Motivation: what R is, its reputation, what is expensive, anvl | 4 | 4:00 |
| 2 | What the front-end can do: formulas, fit, small data | 3 | 3:30 |
| 3 | How it works: pipeline, tracing, IR, rules, gradient, backends, lowering (three 7-step animations) | 27 | 11:30 |
| 4 | Runtime: pjrt, custom calls | 2 | 2:30 |
| 5 | Why IREE: shapes in statistics, BO, workarounds, graph breaks, precision, targets, compile time | 10 | 9:00 |
| 6 | Been here before: JAX on IREE, open issues, anvl on IREE, PJRT | 5 | 5:30 |
| 7 | Open questions: who checks the shapes, which IR | 3 | 3:30 |
| 8 | Plan, questions, thanks | 3 | 1:30 |
|   | **Total** (each animation step counted) | **58** | **~41:30** |

The weight is on parts 3, 5, 6 and 7. The 21 step-through slides are meant
to be clicked through in about 10 seconds each. This is about 16 minutes
over the assumed 25-minute slot; section 9 lists the cuts, and the slot
length is still to be confirmed (section 10). All parts are built in
`index.qmd`. The deck also has a pipeline overview opening part 3, a
lowering step-through after 3.6, and an "Additional Slides" section at the
end (what tracing cannot see, the dispatcher, the gradient in StableHLO,
gotchas, why not just C++, what changes in the frontend). Benchmarks and the
test-suite pass rate are TODO callouts.

## 5. Slide-by-Slide Outline

Each slide lists its **content**, the **visual**, and **notes** for what to
say. Headings are working titles.

### Part 0: Opening

#### 0.1 Title

- Title, name, affiliation (LMU Munich, MaRDI), link to `github.com/r-xla`.
- **Say:** one sentence: "I build a JAX-inspired modeling toolbox for R, and
  I'm here because I think IREE can solve its biggest problem."

### Part 1: Motivation

#### 1.1 What Is R

- **Content:** the language of statistics and data analysis; think of it as
  Python, less rigorously defined: interpreted, dynamically typed,
  copy-on-modify, column-major. What people do with it:
  data wrangling (most of the time), plotting, statistical models (`lm()`,
  `glm()`, mixed models, survival, Bayesian models), reports (the deck itself
  is a Quarto document with live R output).

#### 1.2 Warts and All

- **Content:** R's reputation, jokingly: it has warts (`1:length(x)` counts
  down for an empty `x`, shown live), and nobody would call it the most
  beautifully designed language. But the ecosystem works: CRAN checks every
  new package version against all packages that depend on it, so the latest
  versions of 20,000+ packages work together. No dependency hell (sorry,
  Python).

#### 1.3 What Gets Expensive

- **Content:** R is fast enough for much of statistics. Expensive are
  Bayesian modeling (MCMC: thousands of gradient evaluations; variational
  inference), neural networks, large data (implementations fine for small `n` that do not
  scale, `O(n³)` Gaussian processes), and repetition (bootstrap,
  cross-validation, simulation studies). Today's answer: C++ or Stan.

#### 1.4 Why Not Just C++?

Moved to the "Additional Slides" section at the end of the deck.

- **Content:** a C++ implementation fixes the menu (a GP package offers the
  kernels its author wrote; anything else means changing the C++ code or an
  R callback per matrix entry, slow and without gradients) and fixes the
  hardware (CPU; a GPU means a second implementation). A JIT compiles
  library code and user code into one program, for the device at hand, and
  differentiates through both.
- **Visual:** live example: package code `gp_nll(kernel)` (Cholesky,
  triangular solve) and a user-defined locally periodic kernel; the
  gradient with respect to all hyperparameters.

#### 1.5 anvl in One Slide

- **Content:** a JAX-inspired modeling toolbox for R (not "JAX for R").
  `jit()` compiles, `gradient()` differentiates,
  same code runs on CPU and CUDA (Metal experimental). The r-xla stack:
  `anvl` (tracing + transformations) → `stablehlo` (IR builder) → `pjrt`
  (runtime) → plugin.
- **Visual:** the stack diagram (`openxla-stack.tex`), adapted: anvl on top,
  StableHLO, PJRT, XLA. Leave an empty dashed box next to XLA labelled
  "IREE?" to plant the question.
- **Say:** everything above PJRT is ours, written in R and C++. Everything
  below is OpenXLA, which is the part we want to swap.

### Part 2: What the Front-End Can Do

#### 2.0 The Running Example: Logistic Regression

- **Content:** the model and its negative log-likelihood as formulas, and
  gradient descent. Jokingly, for the deep learning people: a neural network
  with zero hidden layers.

#### 2.1 A Statistical Algorithm, Compiled and Differentiated

- **Content:** the running example. `nll`, `gradient()`, `jit()`, call it on
  data, get a gradient back. Then the full fit with `nv_while()`.
- **Visual:** code (left), the fitted coefficients (right), and a note that
  this is an illustrative example: in practice, logistic regression is fitted
  differently (`glm()` uses iteratively reweighted least squares).
- **Say:** "This is ordinary R. The user never writes a kernel and never
  writes a derivative."

#### 2.1b Small Data Needs Fast Loops, Too

- **Content:** small data is common in statistics (a clinical trial, a lab
  experiment, one group of a per-group fit), so each iteration is cheap and
  per-iteration overhead decides. The same fit as a base R loop with a
  hand-derived gradient, timed at render time against the compiled loop
  (warm) and against the first call for a new shape. Measured on a laptop:
  base R ~65 ms, anvl warm ~9 ms, new shape ~110 ms.
- **Say:** warm, the compiled loop wins; but one new shape costs more than
  the whole base R fit. This sets up part 5.


### Part 3: How It Works

The deck opens this part with the pipeline (trace, transform, lower). Lowering
goes through an `AnvlBackend`, the interface that owns arrays, devices and
`jit()`: the `pjrt` backend (StableHLO through a PJRT plugin) and the
experimental `quickr` backend, which transpiles the graph to Fortran.

This is the core of the talk. Everything is shown on the running example.

#### 3.1 Tracing: Running R on Abstract Values

- **Content:** `jit(f)` does not interpret `f` per call. It calls `f` **once**
  with `GraphBox` objects that carry an abstract value (shape + dtype, no
  data). R's S3 dispatch routes `+`, `%*%`, `exp`, … on a `GraphBox` to a
  primitive, and every primitive call appends a statement to the current
  `GraphDescriptor`. Ordinary R (loops, `if` on non-array values, helper
  functions, closures) runs at trace time and disappears from the program.
- **Visual:** the `for (i in 1:3) x <- x + one` example from the earlier talk
  (traces to three adds), *or* the first statements of tracing `nll`. Use
  `trace_fn()` so the output is real.
- **Say:** this is JAX's `jit` model, with operator overloading done by R's S3
  dispatch rather than Python's dunder methods. Point out what the audience
  will recognise: tracing is partial evaluation with respect to everything
  except array values.

#### 3.1b Tracing, Step by Step (7 slides)

- **Content:** `nll` traced one expression at a time. Left: the R code with
  the expression being evaluated marked, the R variables bound to boxes so
  far (`eta` → `%1`), and one sentence on what happened (S3 dispatch to a
  primitive, statement appended). Right: the graph built so far, with the
  statement this step added marked.
- **Visual:** generated from the real printed graph at render time.
- **Say:** click through quickly; about 10 seconds per slide.

#### 3.2 What Tracing Cannot See

Moved to the "Additional Slides" section at the end of the deck.

- **Content:** a traced value has no data, so `if (x > 0)` fails at trace
  time. Data-dependent control flow must be staged explicitly: `nv_if()`,
  `nv_while()`, `nv_ifelse()`. These trace their bodies into **sub-graphs**
  (closed regions: whatever they read from the enclosing graph becomes an
  operand). `static` arguments are the other escape hatch: they become part
  of the cache key and are ordinary R values during tracing.
- **Visual:** the error message from `jit(function(x) if (x > 0) x else -x)`,
  next to the `nv_if()` version.
- **Say:** the cache key is (input tree structure, dtypes, **shapes**, static
  values). Remember "shapes": it comes back in part 5.

#### 3.3 The IR: `AnvlGraph`

- **Content:** anvl's own IR, deliberately not StableHLO.
  - A graph is a list of `GraphStatement`s in SSA form:
    `(primitive, inputs, params, outputs)`.
  - Values are `GraphValue`s with an abstract value (shape, dtype);
    constants are `GraphLiteral`s or captured arrays.
  - `in_tree` / `out_tree` describe how nested R lists flatten into the
    graph's inputs and outputs (pytrees in JAX terms).
  - Control-flow primitives carry closed sub-graphs as parameters.
  - Side effects (the RNG state) are threaded explicitly as inputs and outputs.
- **Visual:** the printed `AnvlGraph` of `nll`, with callouts on one
  statement, the inputs and the output.
- **Say:** why our own IR rather than tracing straight into StableHLO:
  transformations need primitive-level rules (derivatives, lowering to
  several backends), which is easier on a small IR we control, written in R.

#### 3.4 Primitives Are Bundles of Rules

- **Content:** each of ~100 primitives is an `AnvlPrimitive` with one rule per
  interpretation: `stablehlo` (lowering), `reverse` (VJP), `quickr` (a second
  backend that transpiles to Fortran). The ~160 user-facing `nv_*` functions
  compose primitives and inherit every rule for free.
- **Visual:** `names(attr(prim_add, "primitive")$rules)` output, and a
  two-line reverse rule for `mul`.
- **Say:** this is the design that makes transformations compose.

#### 3.5 Transformations Are Graph Rewrites

- **Content:** `gradient(f)` takes a graph and returns a graph:
  1. requirement analysis: which values need gradients, and which forward
     values the backward pass needs;
  2. replay the forward statements into a new descriptor;
  3. walk the statements backwards, applying each primitive's `reverse` rule
     to accumulate cotangents.

  The result is an ordinary `AnvlGraph`, so it can be jitted, differentiated
  again, or passed through cleanup passes (dead constants, inlining scalar
  constants). Forward mode and `vmap`-style batching are planned as further
  graph → graph passes.
- **Visual:** `three-transformations.svg` (trace → transform → lower),
  then the printed gradient graph of `nll` next to the forward graph.
- **Say:** the transformation happens *above* StableHLO, so any backend,
  including IREE, inherits differentiation without doing anything.

#### 3.5b The Backward Pass, Step by Step (7 slides)

- **Content:** the same visualization for `gradient()`. Left: the forward
  graph with the statement whose reverse rule is applied marked, the
  cotangents so far (`%5` ← `%7`, ...), and one sentence per rule. Right:
  the gradient graph growing, the replayed forward pass folded into one
  line, the statements of this step marked.
- **Visual:** generated at render time; the render fails if the gradient
  graph anvl produces no longer matches the steps.

#### 3.5c One IR, Several Backends

- **Content:** anvl differentiates **its own IR, in R, before lowering**.
  The alternative is to differentiate the compiled program: Enzyme does AD
  on MLIR (as Reactant.jl uses it, on StableHLO). That works for any
  producer of StableHLO, but ties AD to MLIR. Because anvl's
  transformations run above the backend, every `AnvlBackend` inherits them:
  - `pjrt`: StableHLO through any PJRT plugin (XLA today, IREE next);
  - `quickr`: the same graph transpiled to Fortran, with no MLIR at all.

  Separating the *what* (the program, transformations) from the *how* (the
  compiler and runtime) is common now:
  - **JAX**: transformations on jaxprs; lowering to StableHLO for any PJRT
    plugin.
  - **Burn** (Rust): one `Backend` trait; CUDA, ROCm, Metal, Vulkan and
    WebGPU through CubeCL, plus CPU backends.
  - **GoMLX** (Go): XLA through PJRT, a pure Go backend, ONNX Runtime.
- **Visual:** the same `gradient()` call run on both backends
  (`with_backend("quickr", ...)`, as in the earlier talk).
- **Say:** this is why swapping XLA for IREE is a backend question for us,
  not a rewrite: the gradient does not care who compiles it.

#### 3.6 Lowering to StableHLO

- **Content:** each statement's `stablehlo` rule emits ops through our
  `stablehlo` R package: an IR builder written in R that implements the
  spec's type inference, prints MLIR text, and uses 0-based indexing
  (anvl is 1-based; the lowering rules translate). Some ops are CHLO
  (`lgamma`, `digamma`, `erf_inv`, `top_k`, …); a few are `custom_call`s
  (part 4).
- **Visual:** the StableHLO text for `nll`, cropped to `func.func @main` and
  the `stablehlo.dot_general`. All types are static:
  `tensor<1000x5xf64>`.
- **Say:** this is the boundary to the compiler, and the only thing a
  StableHLO-consuming backend has to accept. Point at the static
  `1000x5` and say "this is the problem".

### Part 4: Runtime

#### 4.1 pjrt: R Bindings to the PJRT C API

- **Content:**
  - The `pjrt` R package talks to the **PJRT C API** only: it loads a plugin
    (`GetPjrtApi`), creates a client, compiles StableHLO
    (`PJRT_Client_Compile`), moves buffers (`BufferFromHostBuffer`, events
    for async readback) and executes (`PJRT_LoadedExecutable_Execute`).
  - A native dispatcher in C++ keeps the per-call path out of the R
    interpreter: it hashes the call's input tree, dtypes, shapes and static
    values into a key, looks up an LRU cache of loaded executables, and
    executes. A miss calls back into R to trace, lower and compile.
  - Eager mode is the same path: every primitive is itself jitted, so eager
    operations are small cached executables.
  - The plugins are prebuilt and downloaded (`pjrtbackend`): XLA CPU and
    CUDA; Metal experimental.
- **Visual:** a sequence diagram: R call → dispatcher (hash → hit/miss) →
  [miss: trace → lower → `PJRT_Client_Compile`] → `Execute` → buffers.
- **Say:** the C API is what makes the plugin swappable: in principle
  nothing in anvl needs to change for IREE.

#### 4.2 Custom Calls: Where Portability Breaks

- **Content:** StableHLO has `cholesky` and `triangular_solve`, but no QR,
  LU, SVD or symmetric eigendecomposition. anvl lowers its factorizations to
  `stablehlo.custom_call` (`potrf`, `geqrf`, `orgqr`, `lu`, `svd`, `eigh`,
  plus `print_tensor` for debugging), with handlers registered through the
  **PJRT FFI extension**: LAPACK on CPU, cuSOLVER on CUDA. JAX does the
  same. IREE lowers none of these: not the two StableHLO ops (#10816,
  #11018), and of all `custom_call` targets only
  `ProductOfElementaryHouseholderReflectors`; any other target fails input
  legality.
- **Visual:** one `custom_call @geqrf` line from the lowered IR, and a
  table: op → CPU handler → CUDA handler → IREE: ?
- **Say:** for statistics this is not optional; QR and Cholesky are the
  workhorses of regression. Plant the question here, and come back to it
  with the options in 7.4, after the history in part 6 has shown that IREE
  hit the same problem with JAX in 2023.

### Part 5: Why IREE

#### 5.1 One Compile per Shape

- **Content:** XLA compiles for static shapes. StableHLO can express `?`
  dimensions, but the XLA CPU PJRT plugin rejects them: "Unbounded dynamism
  is disabled for instruction: `f32[?]` parameter(0)" (tested Oct 2026;
  bounded dimensions fail too). JAX refines shapes to static ones itself
  before it calls `Compile`. So a new `(n, p)` means a new trace, lowering
  and compile; the cache only helps when shapes repeat.
- **Visual:** the live demo from the earlier talk (`message("compiling for
  ...")` triggers on each new shape), with the cache size after each call.
  Then one number: the HMC compile costs ~0.2 s per shape vs ~2 ms per
  warm call for 100 draws (`../hmc-results.csv`).
- **Say:** 0.2 s is fine once. It is not fine thousands of times.

#### 5.2 In Statistics, Shapes Vary All the Time

- **Content:** where a varying `(n, p)` comes from, each with a one-line
  example:
  - **Sequential designs:** Bayesian optimization refits a Gaussian process
    after every evaluation, so `n` grows by one per step (5.3).
  - **Cross-validation:** fold sizes differ when `k` does not divide `n`.
  - **Model selection:** best-subset or stepwise regression fits models
    with `p = 1, 2, …` covariates, each a new shape.
  - **Per-group fits:** the same model fitted to each hospital, gene or
    country, each with its own `n`.
  - **Simulation studies:** a grid over `(n, p)` is the experimental design
    itself.
  - **Library code:** a package author jits a function once; users call it
    with shapes the author never sees. Unlike training a network, there is
    no "the" shape.
  - **Interactive use:** each analysis step changes the data.
- **Visual:** a plot of total time vs number of distinct shapes for the
  running example: XLA (compile per shape) vs warm execution only. The XLA
  line grows linearly with the number of shapes and dominates. **To be
  produced** (section 7).
- **Say:** in ML, the shape is a hyperparameter you fix once. In statistics,
  it is part of the data.

#### 5.3 Example: Bayesian Optimization

- **Content:** minimize an expensive black-box function `f`. Each step:
  1. fit a Gaussian process to the `n` points seen so far: an `n × n`
     kernel matrix, its Cholesky factor, and the gradient of the marginal
     likelihood with respect to the kernel's hyperparameters
     (`gradient()` through `nv_chol()` and triangular solves);
  2. maximize the acquisition function (e.g. expected improvement) with
     gradients;
  3. evaluate `f` at the new point, so `n` becomes `n + 1`.

  With XLA, **every step is a new shape**, so both programs are retraced
  and recompiled at every step. On a cheap-to-evaluate `f`, compilation
  dominates the run.
- It brings together everything this talk is about: a dynamic axis, a
  custom call (Cholesky, 4.2), gradients through it, and `f64` (kernel
  matrices are ill-conditioned, and an `f32` Cholesky fails early).
- **Visual:** time per BO step against `n`, for XLA (compile every step),
  XLA warm (the lower bound), and IREE with a dynamic `n` once that works.
  **TODO:** benchmark (section 7).
- **Say:** the padding workaround exists here too (pad the kernel matrix
  with an identity block), but it has to be derived by hand for this model.

#### 5.4 Workarounds Are Not Good Enough

- **Content:**
  - **Pad and mask** with exponential bucketing (pad `n` to the next power
    of two):
    - padding is not neutral: zero rows are fine for `X'X`, but a padded
      Cholesky or `solve` needs identity padding, a padded mean needs the
      true `n`, padded likelihoods need masks;
    - every library author has to get this right for every algorithm;
    - up to 2× wasted compute, still `log2(n_max)` compiles per axis, and
      two axes (`n` and `p`) multiply the bucket count;
    - all in all, not very user friendly.
  - **XLA's bounded dynamism:** a dimension declared with an upper bound
    (`tensor<?x3xf64, #stablehlo.bounds<1024, ?>>`), compiled once; XLA pads
    to the bound internally (`PadToStatic` / `SliceToDynamic`). This is what
    drives dynamic shapes in TorchTPU. But work and memory are at the bound,
    a bound must be known in advance, and the XLA CPU plugin anvl uses
    rejects it ("Custom call target PadToStatic is not implemented", tested
    Oct 2026).
  - Neither handles data-dependent output shapes (`which()`, `unique()`,
    boolean subsetting): 5.4b.
- **Visual:** the pad-and-mask code, with the `solve()` case shown wrong
  under zero padding; one line on bounded dynamism.
- **Say:** this is what dynamic shapes in the compiler would remove.

#### 5.4b Boolean Subsetting: Graph Breaks

- **Content:** `x[x > 0]` has a shape that depends on the data, so it
  cannot be inside one static program. It can be done with a **graph
  break**: split the program where the shape becomes data-dependent.
  1. Program 1 computes the mask and its count `k = sum(x > 0)`.
  2. The host reads `k` back: a synchronization point, the pipeline stalls.
  3. Program 2 is traced and compiled for shape `k`, and runs.

  Program 2's shape depends on the output of program 1, so **every distinct
  `k` is a new compile**. In statistics this is routine: dropping outliers
  and refitting, complete-case analysis, subsetting to a group, keeping
  observations above a threshold. With a compiler that handles dynamic
  shapes, program 2 compiles once for `?`, or the whole thing is one
  program with a data-dependent shape inside.
- **Visual:** a diagram (section 6): program 1 → `k` read back to the host
  → program 2 compiled for `[k]`, with a fan of program-2 variants
  (`k = 812`, `k = 790`, …) to show the cache growing with the data.
- **Say:** this is the "graph break" of `torch.compile`; the cost is the
  recompile, not the break. The examples from statistics (outliers,
  complete cases, groups, thresholds) go in the speaker notes of the live
  demo slide.
- **Loops, too:** a vectorized `nv_while()` that iterates every element until
  it meets a stopping criterion (a Newton solve per observation, a sampler
  per chain) runs until the last element is done, because the loop state has
  a static shape. With dynamic shapes, the body could drop the finished
  elements and return a smaller state in each iteration.

#### 5.5 CPU and Precision Matter

- **Content:** what statistical users run on:
  - laptops, CPU servers and HPC nodes without GPUs; Apple Silicon is common
    among R users;
  - many small and medium problems, where compile time and dispatch
    overhead matter as much as peak throughput.

  And they need **precision**: `f64` is the default in statistics, not an
  option.
  - Log-likelihoods are sums over many observations; `X'X` and kernel
    matrices are ill-conditioned; tail probabilities (`pnorm(-30)`) and
    MCMC acceptance ratios live where `f32` has no digits left.
  - Results are compared against `f64` references (R's `glm()`, Stan).
  - anvl's distribution functions are being reworked to `f64` machine
    precision in the tails; that is only worth it if the backend keeps it.

  What this asks of IREE: no silent demotion of `f64`, a complete `f64`
  math library (`f64` `exp`, `log`, `floor`, `round` only reached the
  embedded CPU math library in Sep 2026, #24953), accurate transcendental
  approximations (#12903 is still open), and fast `f64` vectorized code on
  CPU.

  XLA also flushes subnormals to zero, presumably for performance
  (`1e-200 * 1e-110` is 0 in anvl, `1e-310` in R; anvl#540). That may not be
  necessary any more: subnormals are only slow on Intel CPUs (Daniel
  Lemire's benchmark, x.com/lemire/status/2099845112469635410).

  What IREE offers here: dynamic shapes and LLVM-CPU codegen, which handles
  `f64` (but demotes it to `f32` by default; anvl must pass
  `--iree-input-demote-f64-to-f32=false`). Beyond CPU, targets XLA does not
  have for us:
  - **Metal** (stable in IREE) and **Vulkan**, but both `f32`-only for our
    purposes: Metal has no `f64` at all. Useful for `f32` workloads such as
    HMC with many chains, not for `f64` regression. There is demand: Apple's
    own `jax-metal` plugin has had no release since 0.1.1 (Oct 2024), and
    the community stepped in with `jax-mps` (github.com/tillahoffmann/jax-mps,
    a PJRT plugin on top of MLX, releasing regularly; 0.11.1 in Oct 2026). A
    well-supported Metal target in IREE would serve every PJRT frontend on
    Apple Silicon at once.
  - **WebAssembly**: interesting because **webR** already runs R in the
    browser. IREE's web support is being rebuilt (new freestanding wasm32
    runtime and an experimental WebGPU target since May 2026); there is no
    supported Emscripten build, which webR would need.
- **Visual:** a target matrix: rows = CPU, CUDA, ROCm, Metal, Vulkan, WASM;
  columns = XLA via PJRT / IREE (no `f64` column). ROCm is on both sides
  (XLA has a ROCm PJRT plugin).
- **Say:** "accelerators" is the wrong frame for this audience of users. A
  good CPU backend with dynamic shapes is worth more to them than a faster
  GPU backend.

#### 5.6 Slower Compiles, but Fewer of Them

- **Content:** IREE compiles more slowly than XLA today: this showed while
  running anvl's test suite on it, where every test is a fresh compile.
  **TODO:** quantify (section 7). Dynamic shapes partly offset this. For
  `k` distinct shapes, roughly:
  - XLA: `k × (compile_XLA + run)`;
  - IREE, dynamic: `compile_IREE + k × run`;

  so IREE wins once `k` exceeds about `compile_IREE / compile_XLA`, and in
  the sequential settings of 5.2 and 5.3, `k` is in the hundreds. A program
  that does not depend on the shape can also be cached on disk, or compiled
  ahead of time and shipped with an R package.
- **Visual:** total time against the number of distinct shapes, three
  lines: XLA, IREE with static shapes, IREE with dynamic shapes; mark the
  crossover.
- **Say:** compile time is still a first-class metric for this audience
  (interactive use, small problems), so it belongs in the asks. The
  exponential compile time found by the test suite (#24886) was fixed
  within a month.

### Part 6: Been Here Before

Most of the audience joined IREE after 2023. This part tells them what was
tried then, what was left open, and what happened when anvl's test suite
ran on IREE in 2026. All issue numbers are `iree-org/iree` unless marked
`jax#`.

#### 6.1 JAX on IREE, 2021–2023

- **Content:** a timeline.
  - **2021–22:** JAX has an experimental in-tree IREE backend
    (`JAX_PLATFORMS=iree`), suggested as a route to Apple GPUs (jax#8074).
  - **2022:** JAX's dynamic-shapes prototype ("djax") executes **through
    IREE, because XLA cannot**: a batch-size-polymorphic MLP runs, including
    autodiff (jax#11268, jax#11313). Its design puts tracers for axis sizes
    inside abstract values' shapes, and has to decide when two sizes are
    *symbolically equal*: the same problem as 7.1 and 7.3.
  - **Feb–Jun 2023 (OpenXLA):** JAX runs on IREE through
    `openxla-pjrt-plugin`. Triage of the JAX test suite: 1,495 failing
    compilations in February, mostly `bf16`, `complex` and zero-size
    tensors; 123 left by April (#12070). Epics for custom calls (#12410),
    collectives and `pmap` (#12930, #13253), benchmarking (#13010).
  - **Jul 2023:** JAX removes its in-tree IREE integration; JAX-on-IREE
    should go through the PJRT plugin (jax#16491).
  - **Sep 2023:** the plugin moves into the IREE repository
    (`integrations/pjrt`); `openxla/openxla-pjrt-plugin` is gone.
  - **2024–25:** IREE moves to `iree-org` and joins LF AI & Data (May 2024).
    The plugin gets community fixes (#19202, #19221, #19223, #20907) and a
    last API update in May 2025 (PJRT 0.68, #20930).
  - **2026:** the plugin is still built and tested in CI for CPU and CUDA,
    but has had only mechanical changes since. In the last six months, AMD
    and Roofline AI wrote most of IREE's commits, and none came from
    google.com addresses.
- **Visual:** a horizontal timeline with the four phases.
- **Say:** "You probably weren't around for this. It's worth knowing,
  because anvl hits the same gaps."

#### 6.2 What Was Left Open

- **Content:**

  | Issue | Topic | State (Oct 2026) | Why anvl cares |
  |---|---|---|---|
  | #10816 (2022) | no lowering for `cholesky` | open; prototype commit by rsuderman, Jun 2023 | regression, GPs, multivariate normal densities |
  | #11018 (2022) | no lowering for `triangular_solve` | open | `solve()`, back-substitution after Cholesky / QR |
  | #12410 (2023) | epic: `custom_call` support, "QR, Eigh, etc." via library calls | open; host-side shim prototyped (#13172, PR #13551, never merged); GPU shim, JAX kernel registration, FFI ABI (#13165–#13167) open | anvl's QR, LU, SVD, eigh |
  | #12411 (2023) | JAX host callbacks | open | anvl's `print_tensor` |
  | #12903 (2023) | numerical differences from XLA in elementwise ops | open | anvl targets `f64` precision in distribution tails |
  | jax#15357 (2023) | framework-independent PJRT API to register custom calls, filed from the IREE side | closed Dec 2023: "each backend will define its own API for registering custom calls" | exactly anvl's problem today (7.4) |

  Not relevant to anvl: `bf16`, `complex`, collectives and `pmap`.
- **Visual:** the table, with the anvl column highlighted.
- **Say:** two of the workhorses of statistics, Cholesky and triangular
  solve, have had open issues since 2022 (confirm the current state first,
  section 8).

#### 6.3 anvl's Test Suite on IREE (Sep 2026)

- **Content:** I replaced the XLA CPU PJRT plugin with IREE's and ran
  anvl's test suite, which checks every primitive's lowering against torch;
  for now with static shapes only.
  - Pass rate: **TODO** (numbers from the run).
  - Next (**TODO** for anvl): shape-polymorphic tracing (at least partially)
    and IR with dynamic shapes, then run the tests on IREE again.
  - Issues filed:
    - #24881: the `GatherToTorchIndexSelect` lowering of `gather` drops the
      index clamp, so out-of-bounds indices read arbitrary memory. Open.
      anvl depends on the clamp: it is how indexing avoids runtime errors.
    - #24885: `stablehlo.while`: a loop-carried resource is deallocated on
      the branch into the loop. Fixed in Oct 2026 (2a63818).
    - #24886: LLVMCPU compile time grows exponentially with the def chain
      feeding a data-dependent `tensor.extract`. Fixed in Oct 2026 by an
      LLVM integrate.
- **Visual:** the three issues as cards with their status.
- **Say:** thank the people who fixed them. Offer: anvl's per-primitive
  tests are a ready-made regression corpus for the StableHLO input path, and
  it does not use `bf16` or `complex`, so it is clean signal.

#### 6.4 One Plugin, Many Frontends

- **Content:** PJRT turns M frontends × N backends into M + N. Frontends
  that load PJRT plugins through the C API: **JAX**, **zml** (Zig),
  **GoMLX** (Go, via `go-xla`), **gx** (via GoMLX's bindings), **anvl** (R);
  **Reactant.jl** (Julia) and **EXLA** (Elixir) link XLA directly for CPU
  and GPU but can also load C-API plugins. A solid IREE PJRT plugin gives
  all of them IREE's targets and dynamic shapes without any frontend
  changes.

  What anvl concretely needs from a plugin:
  - `Client_Create`, `Compile` (StableHLO plus a serialized XLA
    `CompileOptionsProto`, which IREE's plugin parses);
  - buffers in and out, row-major layouts, async events;
  - the FFI extension, or an equivalent, for custom calls (4.2).
- **Visual:** the stack diagram with frontends on top, PJRT in the middle,
  plugins below: XLA CPU, XLA CUDA, TPU, Metal, **IREE** (highlighted).
- **Say:** in 2023 the plugin existed to serve JAX. Today it would serve a
  whole list of frontends. It is still in the tree and in CI, but stuck at
  PJRT API 0.68 (XLA is at 0.117). Ask who owns it.

### Part 7: Open Questions

#### 7.1 Who Checks the Shapes?

- **Content:** with dynamic shapes, `nll`'s matmul becomes
  `X : tensor<?x?xf64>`, `beta : tensor<?xf64>`. The program is only
  correct if `dim(X, 1) == dim(beta, 0)` and `dim(X, 0) == dim(y, 0)`. At
  trace time anvl *knows* these constraints. In StableHLO they cannot be
  written as types. The spec: "Constraints will be verified statically if
  possible, otherwise they are deferred to runtime and mismatches will
  result in undefined behavior", and "Generating correct code is the
  responsibility of the producer."

  Who could check them, and what happens today:

  | Where | How | Today |
  |---|---|---|
  | Frontend (R) | anvl emits a host-side check per call | fast, good error messages; every frontend reimplements it; an imported program carries no checks |
  | In the IR | JAX's `custom_call @shape_assertion`; `shape.cstr_*`; `cf.assert` | JAX evaluates its assertions at *compile* time, after refining shapes, and IREE's StableHLO input **drops** them; `shape.cstr_*` is illegal input to IREE (from source, untested); `cf.assert` is stripped at O1 and above |
  | Compiler at the ABI boundary | IREE's `hal.buffer_view.assert` on entry | checks encoding, dtype, rank and static dims; for a dynamic dim it compares the buffer with itself, so nothing is checked |
  | PJRT plugin | checks before `Execute` | needs metadata the IR cannot carry |
  | Nobody | undefined behaviour | the current state: out-of-bounds reads, wrong results, no error |

- **Visual:** the dynamic StableHLO signature, with the two implicit
  equalities drawn as arcs between dimensions; then the table.
- **Say:** R users expect `"ncol(X) must equal length(beta)"`, not a
  segfault, and certainly not a silent wrong answer.

#### 7.2 Constraints Are Also Information

- **Content:** the equalities are not just for checking. A compiler that
  knows `dim(X, 1) == dim(beta, 0)` can form dispatches across ops with
  the same iteration space, drop redundant dimension queries, and pick
  tile sizes; ranges and divisibility (`n % 8 == 0`, `p ≤ 64`) help
  vectorisation. So the question is also: **in which IR should a frontend
  write this down so IREE can use it?**
  - StableHLO: `?` and bounds only; no symbolic dimensions and no RFC for
    them. Its tooling assumes specialization: `stablehlo-refine-arguments`,
    `stablehlo-refine-shapes`, and since Nov 2025
    `stablehlo-check-shape-assertions`.
  - torch-mlir's `torch.symbolic_int` / `torch.bind_symbolic_shape`. IREE
    consumes them (`BindSymbolicShapes`, Oct 2024): the upper bound and
    divisibility become `util.assume.int`, and dimensions that share a
    symbol become one value. That is, equalities are *assumed*, never
    checked.
  - `linalg` / `tensor` directly, with `util.assume.int` for ranges and
    divisibility.
  - A symbolic-shape extension to StableHLO.
- **Visual:** the same matmul in two of these forms, side by side.
- **Say:** this is the main thing I want the audience's opinion on.

#### 7.3 What Changes in the Frontend

Moved to the "Additional Slides" section at the end of the deck.

- **Content:** symbolic tracing in anvl:
  - abstract values carry symbolic dimensions (`n`, `p`) instead of
    integers; shape arithmetic in R (`n * p` for a reshape, `n1 + n2` for a
    concatenation) produces symbolic expressions;
  - R control flow that branches on a shape (`if (nrow(X) > 1000)`) cannot
    be decided for a symbolic `n`: it either raises, or becomes a guard that
    specialises the program (as in `torch.compile`'s guards or JAX's shape
    polymorphism errors);
  - the dispatcher's cache key changes from exact shapes to
    (rank, dtype, symbolic pattern), and the cache stops growing with the
    number of shapes;
  - primitives' shape rules must be written over symbolic dimensions.
- **Visual:** the `nll` graph printed with `f64[n, p]` instead of
  `f64[1000, 5]`.
- **Say:** this is our side of the work, and it is tractable because the IR
  and its shape rules are small and ours. JAX's djax (6.1) went down the
  same road in 2022.

#### 7.4 Who Runs the Library Kernels?

Removed from the deck; kept here for the discussion. The question stays on
the "Questions for You" slide.

- **Content:** anvl's linear algebra needs six custom calls (`potrf`,
  `geqrf`, `orgqr`, `lu`, `svd`, `eigh`) plus `print_tensor`. On XLA they
  are XLA FFI handlers registered through the `PJRT_FFI` extension: LAPACK
  on CPU, cuSOLVER on CUDA. The options for IREE:

  | Option | For | Against |
  |---|---|---|
  | IREE's PJRT plugin implements the `PJRT_FFI` extension and calls the registered handlers from the IREE runtime | frontends keep their handlers unchanged; JAX's LAPACK kernels work the same way | host-side only; the XLA FFI ABI belongs to XLA; device kernels (cuSOLVER) need a stream from IREE's HAL |
  | Native lowerings in IREE (`iree_linalg_ext`, or decomposition into `linalg`) | runs on every target; can fuse | a lot of work; numerically delicate (pivoting, SVD convergence); hard to match LAPACK's speed; on Metal only in `f32` |
  | Frontend emits IREE's own extension points (`hal.dispatch.extern`, executable objects and plugins, custom VM modules, `iree_linalg_ext.custom_op`) | per-target tuned kernels; all of these exist today | the frontend emits IREE-specific IR, so StableHLO stops being the contract |
  | Frontend decomposes into plain StableHLO (`while` loops) | portable today | slow; the compiler sees loops, not a factorization |

  With dynamic shapes, this gets harder: a custom call's result shapes and
  LAPACK's workspace size depend on `n`, so they are only known at run time.
- **Visual:** the table; one `custom_call @geqrf` line above it.
- **Say:** in 2023, the answer from the JAX side was that every backend
  defines its own registration API (jax#15357). Ask what the IREE
  community's answer is today, and whether a shared answer for all PJRT
  frontends is possible.

### Part 8: Plan and Asks

#### 8.1 Plan

- **Content:**
  1. **Static, done:** anvl's test suite runs on IREE's PJRT plugin (6.3).
     Next: the remaining failures, and compile time and run time against XLA
     on the statistical benchmarks (regression, HMC, distributions).
  2. **Custom calls:** whichever route 7.4 settles on, starting with
     Cholesky and triangular solve.
  3. **Dynamic:** shape-polymorphic tracing for a subset of primitives; emit
     `?` dimensions and constraints in whichever form we agree on (7.2);
     then run the test suite on IREE again.
  4. **PJRT:** help keep the IREE PJRT plugin tested, so that the swap stays
     configuration rather than code.
  5. **Targets:** Metal and WebAssembly (webR).
- **Asks:** the four questions from section 1, on one slide, left up during
  the discussion.
- **Visual:** the four questions only.

#### 8.2 Thanks

- Links: `github.com/r-xla/anvl`, docs site. anvl's initial design was based
  on JAX's Autodidax tutorial (docs.jax.dev/en/latest/autodidax.html). MaRDI
  acknowledgement.

### Backup Slides

- Full StableHLO dump of the gradient of `nll`.
- A reverse rule in full, and how `rebuild_forward_into` / requirement
  analysis decide what to recompute.
- Side effects: how the RNG state is threaded through traces and loops.
- The quickr backend: the same `AnvlGraph` lowered to Fortran, as evidence
  that the IR is not tied to StableHLO.
- Benchmarks: HMC vs Stan, GELU fusion vs torch (`../gelu-benchmark.csv`),
  CPU threading (`../cpu-threads.csv`).
- Gotchas relevant to compiler people: no `NA` on hardware, index clamping
  instead of runtime errors.

## 6. Diagrams to Produce

| Figure | Source | Notes |
|---|---|---|
| Stack with "IREE?" box | adapt `../openxla-stack.tex` | slide 1.2 |
| Trace → transform → lower | `../three-transformations.tex` | slide 3.5, reuse |
| Dispatcher sequence | new, TikZ | slide 4.1: hit path vs miss path |
| JAX-on-IREE timeline | new, TikZ | slide 6.1: 2021 in-tree backend → 2022 djax → 2023 PJRT plugin → Jul 2023 removal → 2026 anvl |
| M + N PJRT diagram | adapt `../openxla-stack.tex` | slide 6.4: more frontends, IREE plugin highlighted |
| Dynamic signature with constraint arcs | new, TikZ | slide 7.1 |
| Graph break | new, TikZ | slide 5.4b: program 1 → `k` read back to the host → program 2 compiled for `[k]`, with a fan of program-2 variants for different `k` |

Keep the existing style: rounded boxes, Stealth arrows, blue for "ours",
grey for the shared stack.

## 7. Benchmarks to Produce

Write every result to a CSV, as the earlier talk does, so text and numbers
cannot disagree.

- **Compile cost vs number of shapes** (slides 5.2, 5.6): fit the running
  example to `K` datasets with distinct `(n, p)`, `K = 1 … 100`; report
  total time, compile time and warm execution time. XLA now; IREE with
  dynamic shapes once that path works (**TODO**).
- **Compile time, XLA vs IREE, static shapes** (slide 5.6). **TODO: run.**
  Per-program compile time for the running example, HMC and the GP fit,
  both through PJRT on the same CPU, plus the distribution of compile times
  over all programs of anvl's test suite. Report warm run time alongside,
  so that a slower compile is not mistaken for a slower program.
- **Bayesian optimization** (slide 5.3). **TODO: run.** Time per step
  against `n` for a GP-based BO loop on a cheap test function (e.g.
  Branin), `n` from 10 to 200: XLA cold (compile every step), XLA warm, and
  IREE with a dynamic `n` once that works.
- **anvl test suite on IREE** (slide 6.3): pass / fail / skip counts, with
  failures grouped by cause (custom calls, known IREE issues, other).

## 8. Verified Claims

Checked on 2026-10-27 against IREE `69e0d66e40`, XLA `dd678a263b`, JAX
`344b318b45` and StableHLO `44151da3f3`; claim 9 was also tested with the
local `pjrt` package and the XLA CPU plugin. The slides above already use
these results.

| # | Claim | Verdict | Where it is used |
|---|---|---|---|
| 1 | IREE PJRT plugin exists | in-tree (`integrations/pjrt`, since Sep 2023), CI-tested for CPU and CUDA, "experimental"; stuck at PJRT API 0.68 since May 2025 | 6.1, 6.4 |
| 2 | Metal, WASM targets | Metal stable; web support rebuilt (freestanding wasm32 runtime, experimental WebGPU), no Emscripten build | 5.5 |
| 3 | `f64` | works on llvm-cpu but demoted to `f32` by default; Metal and WGSL have no `f64` | 5.5 |
| 4 | `hal.buffer_view.assert` | checks encoding, dtype, rank, static dims; dynamic dims are not checked | 7.1 |
| 5 | torch symbolic shapes → `util.assume.int` | yes, `BindSymbolicShapes` (Oct 2024) | 7.2 |
| 6 | equality constraints in IREE | assumed via torch symbols; `shape_assertion` dropped; `shape.cstr_*` illegal; `cf.assert` stripped at O1+ | 7.1, 7.2 |
| 7 | JAX `shape_assertion` | yes, but evaluated at compile time after shape refinement | 7.1 |
| 8 | StableHLO dynamic mismatches | undefined behaviour; no symbolic-shape RFC | 7.1, 7.2 |
| 9 | XLA via PJRT refines `?` dims | **refuted**: the XLA plugin rejects them; JAX refines before compiling | 5.1 |
| 10 | PJRT and dynamic output shapes | buffers carry their dims; `OutputDimensions` has no "unknown" value | 6.4 |
| 11 | frontends on the PJRT C API | zml, GoMLX, gx: yes; Reactant.jl, EXLA: link XLA for CPU/GPU, load plugins for others | 6.4 |
| 12 | IREE custom kernels and linalg | extension points exist; no QR, LU, SVD, eigh; `cholesky` and `triangular_solve` still unlowered | 4.2, 6.2, 7.4 |

Still open: whether `PJRT_Buffer_Dimensions` may block on a data-dependent
output shape, whether llvm-cpu can produce wasm32 executables today, and a
citable source for TorchTPU relying on XLA's bounded dynamism (5.4).

## 9. Cutting for a Shorter Slot

For 30 minutes: drop 2.2 and 3.4, merge 1.3 into 1.4, keep only one of the
two step-through sequences (tracing, 3.1b, is the one compiler people know
least from R), merge 5.1 into 5.2, merge 6.1 and 6.2 (timeline with the open
issues as annotations), drop 7.3. For 25 minutes, additionally: drop 1.2,
merge 3.1 and 3.2, merge 5.4 into 5.2, merge 5.6 into 5.5. Slides 2.1b,
5.3, 5.4b, 5.5, 6.3, 6.4, 7.1, 7.2 and 7.4 stay: they are why this talk is
given to this audience.

## 10. Decisions for the Author

- Talk length and whether there is a live demo or only rendered output.
- Whether to show Metal at all, given anvl's own Metal support is
  experimental.
- Numbers from the September test-suite run for 6.3, and which IREE PJRT
  plugin build it used.
- How much of the 2023 history to show: two slides (6.1, 6.2) or one.
