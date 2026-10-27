# Review of the IREE Talk Deck

Review of `index.qmd` as rendered on 2026-10-27. Slide numbers are the ones
the deck shows; `14/3` is step 3 of the animation on slide 14. The review
combines three passes: a fact check of every claim (sources below), an
independent read for audience fit and density, and a few measurements made
for this review.

## Summary

- **Factual errors are few but some are visible.** The ones a compiler
  audience will catch: the precision demo shows a range problem, not a
  precision problem (slide 33); "one environment variable" is two, and the
  plugin file name is wrong (23, 39); "shipped with an R package" contradicts
  CRAN policy (35); the TorchTPU claim overstates what TorchTPU does (30).
- **Framing risk:** the claim that deep learning has fixed shapes (notes of
  27) will meet pushback, because IREE's flagship workload, LLM serving, is
  built on dynamic sequence lengths. Credit that work instead.
- **Audience fit:** the most valuable slides for compiler people are 24
  (custom calls), 26/27 (one compile per shape), 32 (graph breaks, shrinking
  loops), 38–39 (history, test suite) and 43–45 (shape constraints, which
  IR). The least valuable are the explanations of tracing, reverse-mode AD
  and the 1:1 lowering, which this audience knows from JAX.
- **Missing evidence:** nothing in the deck shows IREE solving the problem.
  A single experiment (dynamic StableHLO compiled once with `iree-compile`,
  run at several `n`) would answer the first question of the discussion.
- **Length:** about 45–50 minutes as it stands, roughly twice a 25-minute
  slot.

## 1. Fact Check

### Needs a fix

| slide | claim | finding | suggested wording |
|---|---|---|---|
| 4 | "CRAN checks every new package version against all packages that depend on it, so the latest versions of 20,000+ packages work together" | Partly. CRAN re-checks reverse dependencies on submission and asks the maintainer about breakages, but does not block automatically; packages that stay broken are archived. CRAN lists 25,537 packages. ([CRAN policy](https://cran.r-project.org/web/packages/policies.html), [R Journal 2018](https://journal.r-project.org/archive/2018-1/cran.pdf)) | "CRAN re-checks a package's reverse dependencies on every update and archives packages that stay broken, so the current versions of ~25,000 packages pass their checks together." |
| 6 | "~160 user-facing functions over ~100 primitives" | The installed anvl exports 177 `nv_*` functions and 101 `prim_*` primitives. | "~175 user-facing functions over ~100 primitives" |
| 9 | "`glm()` uses iteratively reweighted least squares, a Newton method" | Partly. IRLS is Fisher scoring, which equals Newton only for canonical links (logit is one). | "`glm()` uses IRLS (Fisher scoring; Newton's method for the logit link)" |
| 23 | `PJRT_PLUGIN_PATH_CPU=.../libiree_pjrt.so` | The IREE CPU plugin is built as `pjrt_plugin_iree_cpu.so`. | use the real file name |
| 24 | "The handlers are XLA FFI handlers, registered through PJRT's FFI extension. JAX does the same." | Partly. JAX uses XLA FFI custom calls too, but registers them in-process on CPU (XLA:CPU is linked into jaxlib) and through the PJRT GPU custom-call extension on CUDA. JAX lowers Cholesky to the native `stablehlo.cholesky` by default and only overrides it with `potrf` on CPU and GPU. Expect "why `@potrf` and not `stablehlo.cholesky`?" | "JAX does the same: LAPACK/cuSOLVER as XLA FFI custom calls (in-process on CPU, through a PJRT extension on CUDA)." Prepare an answer for `@potrf`. |
| 29 | "a compiled program cannot raise an error" | True for StableHLO/XLA; IREE can trap at run time (`cf.assert` lowers to `vm.cond_fail` at O0). | "StableHLO has no way to raise an error at run time" |
| 30 | "This is what drives dynamic shapes in TorchTPU" | Partly. `PadToStatic`/`SliceToDynamic` are real (`xla/service/dynamic_padder.h`). TorchTPU (Google/Meta, open-sourced Oct 2026, [github.com/google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu)) builds its dynamic-shape support on bounded dynamism, but it is off by default and `torch.compile` is forced to `dynamic=False`. | "TorchTPU builds its dynamic-shape support on XLA's bounded dynamism (not yet on by default)." |
| 33 | `pnorm(-30)` is 0 in `f32` | Correct output, wrong lesson: 4.9e-198 is outside `f32`'s range (smallest normal ~1.2e-38), so this is a range problem, and statisticians solve it in log space. | Replace with a precision failure. Measured with anvl: the Cholesky factor of a squared-exponential kernel matrix (100 points on [0, 1], length scale 0.3, jitter 1e-8) is **entirely NaN in `f32`** (all 5,050 lower-triangle entries) and correct in `f64`. |
| 33 | XLA flushes subnormals "presumably for speed"; "subnormals are only slow on Intel CPUs" | Flushing is confirmed (`tsl::port::ScopedFlushDenormal` around every execution in `xla/pjrt/cpu/cpu_client.cc`; `xla_cpu_ftz` defaults to true "for backwards compatibility", to match TensorFlow). Lemire's benchmark: Intel ~45–50× slower multiplies; AMD Zen 5 multiplies at full speed but divides ~2.3× slower; Graviton and Apple M4 no penalty. The X link returns HTTP 402 when you're not logged in. | "XLA:CPU flushes subnormals to zero (a TensorFlow legacy). They are mainly slow on Intel (~50× for multiplies); recent AMD and Arm barely care." Link the [blog post](https://lemire.me/blog/2026/09/15/subnormal-floating-point-numbers-are-expensive-on-intel-processors/) instead of X. |
| 35 | "IREE compiles more slowly than XLA today" | Unmeasured (your observation from the test-suite run). The break-even `k > compile_IREE / compile_XLA` assumes dynamic code runs as fast as code specialized to the shape, which compiler people will question. | Keep it marked as an observation until the benchmark exists; add "assuming similar run time". |
| 35 | "cached on disk, or even compiled ahead of time and shipped with an R package: one compile, ever" | Partly wrong. A `.vmfb` is tied to a CPU target and the IREE runtime version (the PJRT plugin compiles for `host`; a generic target is slower). CRAN forbids binary executable code in source packages, so shipping only works outside CRAN (r-universe, GitHub). An on-disk cache in `tools::R_user_dir()` is fine. | "Shape-independent programs compile once per machine and IREE version and can be cached on disk." |
| 37 | Timeline diagram: "Jul 2023: JAX removes its IREE backend; PJRT plugin moves into IREE" | The plugin moved into the IREE repository in Sep 2023, not Jul. | "Jul 2023: JAX removes its IREE backend. Sep 2023: the plugin moves into IREE." |
| 39 | "one environment variable" | Two are needed: the plugin also has to find `libIREECompiler` (`IREE_PJRT_COMPILER_LIB_PATH`; R's pjrt only passes integer create options, so the variable is the way). | "two environment variables" |
| 39 | anvl's test suite "checks every primitive against torch" | Overstated. Primitives are compared against torch where torch covers them, and tested by hand otherwise (anvl's AGENTS.md). | "checks every primitive, mostly against torch" |
| 40 | "gives every frontend IREE's targets and dynamic shapes, with no frontend changes" | Contradicted by the deck itself: custom calls need a registration route (24), and JAX refines shapes to static before compiling (26), so JAX gets no dynamic shapes unchanged. | "with few frontend changes, as far as their StableHLO is portable" |
| 43 | "StableHLO cannot say it" | JAX's `shape_assertion` custom call says it by convention. | "StableHLO has no first-class way to say it" |
| 44 | "`shape.cstr_*` is illegal input"; "`cf.assert` is stripped at O1+"; a dynamic dim "is compared with itself" | From reading IREE's source (Oct 2026), not tested. The authors of that code may be in the audience. | Link each claim to the source line, or say "as far as I can tell from the source", and turn the "nobody" row into a question. |
| 45 | torch-mlir snippet | The syntax is from IREE's test file, but the `min_val`/`max_val` values are made up. | Label it "illustrative". |
| 47 | "Static, done" | Conflicts with the TODO pass rate on 39. | "Static: runs; pass rate on slide 39" |
| notes 27 | "In deep learning, the shape is a hyperparameter you fix once" | IREE's flagship workload (AMD's LLM serving) exports prefill with a dynamic sequence length and a dynamic KV-cache page count, compiled once; IREE has `util.assume.int` and a `torch.export` dynamic-shapes sample. True for training, not for inference. | "In deep learning, dynamic dims are a few bounded axes (batch, sequence); in statistics, every dataset brings a new `n` and `p`." |
| notes 37 | Commit counts by company, "you probably weren't around" | Reads as a comment on Google leaving the project. | Drop the commit counts; keep "this history is worth knowing". |
| backup 55 | "an R callback for every entry of an `n × n` matrix" | Overstated: a C++ package can call a vectorized R function once per matrix. It still goes through the interpreter and gives no gradients. | "an R callback through the interpreter, without gradients" |

### Confirmed

- **IREE and PJRT facts** (checked in the earlier verification pass, source
  commits in `design.md` section 8):
  - The PJRT plugin is in-tree and CI-tested, at PJRT API 0.68 (XLA is at 0.117).
  - `f64` is demoted to `f32` by default.
  - Metal is a stable target and has no `f64`.
  - `hal.buffer_view.assert` does not check dynamic dims.
  - `BindSymbolicShapes` turns torch symbols into `util.assume.int`.
  - IREE's StableHLO input drops `shape_assertion`.
  - The StableHLO spec quote on undefined behavior is exact.
  - The XLA CPU plugin rejects `?` dimensions and bounded dimensions (both tested).
- **Issues and history:**
  - IREE issues #10816, #11018, #12410 ("QR, Eigh, etc."), #12411 and #12903 are open.
  - jax#15357 was closed with "each backend will define its own API for registering custom calls".
  - #24881 is open; #24885 and #24886 are fixed.
  - jax#8074, jax#11268, jax#11313 and jax#16491 say what the deck says.
  - The JAX test-suite numbers from #12070 (1,495 failing in February, 123 left by April) are right.
  - IREE joined LF AI & Data in May 2024.
- **Other projects:**
  - Reactant.jl uses Enzyme-MLIR for AD.
  - Burn has a `Backend` trait with CubeCL backends for CUDA, ROCm, Metal, Vulkan and WebGPU.
  - GoMLX's default backends are XLA and pure Go; its ONNX Runtime backend needs a build tag.
  - `jax-metal`'s last release is 0.1.1 (Oct 2024).
  - `jax-mps` is a PJRT plugin on MLX, last released Oct 23, 2026.
  - The Autodidax link resolves.
- **R:**
  - R's `if` has no dispatch hook: it calls `asLogicalNoNA` in C; tested with S3 and S4 objects.
  - `1:length(x)` gives `1 0` for an empty `x`.
  - R arrays are column-major and copy-on-modify.
- **anvl and pjrt:**
  - The custom-call targets (`potrf`, `geqrf`, `orgqr`, `lu`, `svd`, `eigh`) are XLA FFI handlers on LAPACK (CPU) and cuSOLVER (CUDA).
  - The XLA plugins come from zml's `pjrt-artifacts`.
  - `PJRT_PLUGIN_PATH_CPU`, `AnvlBackend` and the `quickr` backend exist.
  - Every code output and timing in the deck is computed at render time.
- **One opportunity:** anvl's own Metal support (slide 6, "experimental")
  downloads Apple's `jax-metal` 0.1.1, the unmaintained plugin from slide 34.
  Saying so strengthens the case for a Metal target in IREE.

## 2. Audience Fit

The audience are compiler engineers who know JAX's design. Verdicts from the
independent read, with my own view where it differs. Several of these
conflict with things you asked for explicitly (the animations, the formula
slide, "Warts and All"); those are options, not corrections.

| slides | verdict | why |
|---|---|---|
| 3–5 (What Is R, Warts and All, What Gets Expensive) | shorten to one or two slides | Context they need, but only briefly. "Warts and All" is fun; nothing later depends on it. |
| 8 (logistic regression formulas) | cut or merge into 9 | The likelihood formula is for statisticians; the joke can move to 9. |
| 9, 10 | keep, merge 10's table into 26 | 10's point (compile cost vs warm run) belongs with "one compile per shape". |
| 12, 13 | merge 12 into 6; 13 to one line | Tracing on abstract values is JAX's model. |
| 14 (tracing animation) | the reviewer would cut it; I would keep it, shortened to 3–4 steps | The R-specific part (S3 dispatch instead of `__add__`, R variables bound to boxes) is new to them; the rest is not. |
| 15, 16 | keep | 16 (primitives as bundles of rules) is the best design slide. |
| 17, 18 (graph rewrites, backward-pass animation) | cut, keep one sentence on 16 | Textbook reverse-mode AD. The message that matters is "AD happens above StableHLO, so IREE inherits it". |
| 19 (one IR, several backends) | backup | Interesting, but not why they are listening. |
| 20, 21 (lowering, lowering animation) | collapse to one static slide | The 1:1 op mapping is not news. Keep the static `tensor<1000x3xf64>` as the punchline. |
| 23 (pjrt) | shorten, merge with 40–41 | The C API usage matters, the R code less so. |
| 24, 26, 27 | keep | Core of the talk. |
| 28 (Bayesian optimization) | shorten | Replace prose with a three-box loop and the (TODO) plot. |
| 29, 30 (workarounds) | merge into one slide, no code | They know bucketing; a NaN from a singular matrix surprises nobody. |
| 31, 32 (graph break) | merge, code to backup | Keep the shrinking-loop-state point: it is a genuinely new use case for them. |
| 33 (CPU and precision) | keep, trimmed, with the Cholesky demo | `f64` and subnormals are compiler-relevant asks. |
| 34 (targets) | backup, one line on 40 | |
| 35 (slower compiles) | keep only with data | |
| 37, 38 (history) | merge | |
| 39, 40, 43–45 | keep | 43–45 are the most valuable slides of the talk. |
| 47, 48 | merge | |
| 8 section dividers | keep three at most | Each costs a click and a few seconds. |

**Missing, and likely to be asked:**

1. **Evidence that IREE solves the problem.** Slide 39 tests static shapes
   only. Hand-edit the StableHLO of `nll` to `tensor<?x?xf64>`, compile it
   once with `iree-compile`, run it at several `n`, and compare with XLA's
   compile-per-shape. Without it, the first question is "have you tried it?"
2. **Where the per-shape cost goes.** Measured for this review (warm
   session, the gradient-descent fit, `n` ≈ 1000): tracing 7 ms, lowering
   4 ms, XLA compile 44 ms, first call in total 65 ms. Compiling is about two
   thirds of the cost, so dynamic shapes in the compiler address the
   dominant part; one line on slide 26 would pre-empt the question.
3. **Why PJRT instead of IREE's runtime API**, given the plugin is
   unmaintained. Prepare an answer (one runtime layer for all backends,
   M + N, the existing `pjrt` package).
4. **Credit for IREE's existing dynamic-shape work** (torch symbolic ints,
   LLM sequence lengths) when making the case.
5. **How column-major R arrays reach row-major buffers.** The earlier
   "row-major" gotcha turned out not to hold for the current anvl; have the
   correct answer ready.

## 3. Overloaded Slides

Word counts exclude code; a comfortable slide holds about 60–80 words of
prose plus one code block of up to 12 lines.

| slide | content | problem | fix |
|---|---|---|---|
| 33 CPU and Precision | 164 words, 2 code blocks, 4 bullets, 2 paragraphs | four messages on one slide | one demo (Cholesky), three bullets; asks to the questions slide, subnormals to their own line or the notes |
| 28 Bayesian Optimization | 147 words, TODO plot | prose where a diagram would do | three-box loop and the plot |
| 39 Test Suite | 129 words, callout, 4-row table | pass rate hidden in a callout | pass rate large, issue rows below, "next" to the plan |
| 35 Slower Compiles | 119 words, formula, TODO | | one plot with the break-even marked; caching to the notes |
| 10 Small Data | 112 words, code, table | base R code adds little | drop the code, keep the table |
| 27 Shapes Vary | 108 words, 6 bullets, TODO | | four short bullets and the plot |
| 5 What Gets Expensive | 106 words | | three short bullets |
| 38 What Was Left Open | 7-row, 4-column table, 103 words | | four rows (#10816, #11018, #12410, jax#15357), three columns |
| 34 Targets | 7-row table, 86 words | | backup, or table only |
| 44 Who Checks Today | 5-row, 3-column table of long cells | dense | cells of at most eight words; details to notes |
| 24 Custom Calls | code excerpt and 6-row, 5-column table | | drop the code excerpt |
| 29 Pad and Mask | 90 words, 2 code blocks | | one code block or none |
| 15 The IR | graph dump and five claims | | two claims (similar to StableHLO; 1-based axes and composite primitives) |
| 18/7, 21/7 | 44–47 lines of code | not readable at 1050 × 700 | if kept, fold more lines (as the forward pass already is) |

## 4. Time

As it stands: 41 content slide numbers, 8 dividers and 21 animation steps,
about 45–50 minutes. To reach 25 minutes, in order of savings:

1. Cut the backward-pass animation, collapse the lowering animation to one
   slide, shorten the tracing animation to 3–4 steps (−6 min).
2. Cut or merge 4, 8, 13, 17 and 20; move 19 to backup (−5 min).
3. Merge 10 into 26, 29 with 30, 31 with 32 (−4 min).
4. Merge 3 with 5 and 12 into 6; fold 23 into 40; move 34 to backup
   (−4 min).
5. Merge 37 into 38 and 47 into 48; keep three dividers (−2 min).
6. One message each for 28, 33 and 35 (−1.5 min).

## 5. Next Steps

1. Apply the fixes in section 1; most are one-line wording changes.
2. Decide the slot length, then pick from section 4.
3. Produce the missing evidence: the dynamic-shape experiment with
   `iree-compile`, the pass rate of the test-suite run, and the compile-time
   benchmark (all TODO in the deck).
