# EuroBioC poster for anvl — design

## Purpose

A conference poster presenting the `anvl` R package at the European Bioconductor
Conference (EuroBioC). The poster's focus is **what anvl can do** for end users
(a general-purpose pitch), with one concrete bioinformatics-relevant application
(randomized SVD). Built in Typst.

Source content: `eurobioc-poster/abstract.md` (the submitted abstract) and
`talks/bioc-gpu-2026.qmd` (a related talk, the source of accurate API details and
the "three code transformations" diagram).

## Format & tooling

- **Size/orientation:** A0 portrait (841 × 1189 mm).
- **Engine:** Typst.
- **Fonts (accessibility spec):** Arial (present at
  `/System/Library/Fonts/Supplemental/Arial.ttf`).
  - Poster title: Arial 82 pt Bold (60 pt minimum), 12 pt space after.
  - Authors and affiliations: Arial 36 pt.
  - Body text: Arial 32 pt minimum.
  - Code blocks: a monospace font (JetBrains Mono, available in `~/Library/Fonts`)
    counts as "or equivalent"; sized so code remains legible (≥ 28 pt).
- **Color scheme:** navy (`#1a3a5c`) for the header, footer, and headings;
  amber/gold (`#fdf2e0` fill, `#e8c98a` border) for the highlighted SVD band;
  light gray (`#f0f3f6` fill, `#d5dde5` border) for standard section bands;
  a muted blue (`#eaf0f6` / `#b9cbdc`) for the slim "how it works" band;
  a muted red (`#fbe9e7` / `#e0a89c`, text `#9c3a28`) for scope/caveat callouts;
  dark (`#2b2b3a`) for code blocks.

## Layout

Banded layout — mostly full-width stacked bands for an accessible top-to-bottom
read, with one two-column row. Top to bottom:

1. **Header** — navy block. The anvl hexagon logo sits at the left, the title
   block (title + authors) is centered in the remaining width.
   - Title: *"anvl: accelerated array computation for R"*.
   - Authors: Sebastian Fischer¹, Daniel Falbel, Martin Binder¹, Bernd Bischl¹.
   - Affiliation: ¹LMU Munich, Munich Center for Machine Learning (MCML).
     Daniel Falbel has no affiliation. Bischl is last (senior author).

2. **Motivation** — full-width light band. The core pitch only: anvl brings
   compiled, GPU-accelerated array computing to the R community; write array
   programs in plain R, compiled under the hood via the OpenXLA stack (also
   behind JAX, TensorFlow, PyTorch-XLA); anvl aspires to be for R what JAX is
   for Python. (The "why anvl" advantages live in section 4, not here.)

3. **Capability row** — two equal columns, each with a short code snippet:
   - **① Execution: eager & JIT** — Eager mode (default): each operation runs
     immediately; good for interactive use and debugging. JIT mode: `jit()`
     compiles a whole function into one fused executable → speedups.
   - **② Automatic differentiation** — `gradient()` returns reverse-mode
     gradients of plain R functions; no manual derivation.

4. **What anvl can do** — full-width light band. Brief API overview in two
   sub-columns:
   - *~140 array operations* — construction, arithmetic, linear algebra
     (matmul, qr, svd, chol, solve), reductions & statistics, random numbers,
     JIT-able control flow (`nv_while`).
   - *Build, in plain R* — neural networks, matrix factorizations, custom
     optimizers, gradient-based estimators, Monte Carlo samplers.
   - **"Why anvl"** line (the advantages, moved here from Motivation): GPU
     acceleration without leaving R; simplified CUDA install (just a driver);
     JIT compilation sets it apart from `{torch}`; written in R, so easy to
     extend and contribute to.
   - Highlighted **scope caveat** line: anvl currently targets **dense** arrays;
     sparse matrices are not well supported yet. Best fit: dense data, regular
     shapes, repeated/batched computation.

5. **How it works** — slim full-width band. The `three-transformations` diagram
   (R code —trace→ AnvlGraph IR —lower→ Executable, with a transform/gradient
   self-loop) plus the one-line label Trace / Transform / Lower. Kept small —
   the poster's focus is capabilities, not internals.

6. **Application: randomized SVD for bioinformatics** — full-width amber
   highlight band. Brief algorithm sketch, anvl code for randomized SVD, and a
   benchmark figure. The benchmark figure is a **clearly-marked placeholder**
   (the user supplies real data later). Includes a short Bioconductor-specific
   note: sparse single-cell / genomics matrices would need densifying first;
   native sparse support is future work.

7. **Getting started** — full-width light band with a QR code on the right.
   - Install from r-universe (CPU works out of the box; NVIDIA GPU via the
     `cuda12.8` package).
   - Runs on: CPU — Linux, Windows, Apple Silicon; NVIDIA GPU — Linux x86_64.
   - Status: early-stage, not yet on CRAN — feedback and contributions welcome.
   - Links: `github.com/r-xla/anvl`, `r-xla.github.io/anvl`.
   - QR code encodes `https://github.com/r-xla/anvl`.

8. **Footer** — slim navy block. The MaRDI logo sits at the left on a small
   white panel (so the logo's blue "M" stays legible against navy), followed by
   the acknowledgment: "Supported by MaRDI — the Mathematical Research Data
   Initiative (NFDI, DFG grant 460135501)."

## Assets

- `eurobioc-poster/poster.typ` — the poster source (single file).
- `eurobioc-poster/abstract.md` — already created; the submitted abstract.
- `eurobioc-poster/assets/three-transformations.svg` — copied from `talks/`.
  Embedded via Typst `image()`.
- `eurobioc-poster/assets/anvl-logo.svg` — copied from
  `~/r-xla/anvl/man/figures/logo.svg`. Placed in the header.
- `eurobioc-poster/assets/mardi-logo.png` — copied from
  `~/Downloads/mardi-logo.png`. Placed in the footer on a white panel.
- Placeholder benchmark figure — drawn natively in Typst (a bordered box
  labelled e.g. "Benchmark figure — data pending") so no external image file is
  needed; replaced with a real figure/image later.
- QR code — generated with a pure-Typst package (`tiaoma`) so there is no
  external dependency or pre-generated image.

## Code snippets

All R snippets are illustrative and must be verified against the `../anvl/`
package source and the `talks/bioc-gpu-2026.qmd` examples during implementation.
Confirmed API from the talk: `jit(f)`, `gradient(f, wrt = ...)`, `nv_array`,
`nv_scalar`, `nv_while`, `nv_logistic`, `%*%`, and `nv_*` operations such as
`nv_matmul`, `nv_qr`, `nv_svd`. The randomized-SVD snippet must use operations
anvl actually supports; the exact return shapes of `nv_qr` / `nv_svd` are to be
checked in the anvl source.

## Open items resolved during implementation

- Verify exact anvl public API names and signatures against `../anvl/` source.
- Confirm the randomized-SVD algorithm is expressible with anvl's supported
  operations; keep the snippet faithful to the real API.
- Confirm the `tiaoma` Typst package is the right choice for the QR code;
  fall back to a pre-generated QR PNG (`qrencode`) if it is unavailable.

## Out of scope

- Real benchmark data for the SVD section (user provides later).
- Printing / PDF-export pipeline beyond `typst compile` producing a PDF.
