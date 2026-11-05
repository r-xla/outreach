# anvl: accelerated array computation for R

The {anvl} package is an accelerated array computation framework that brings the same JIT compilation and automatic differentiation technology behind modern deep learning to the R community.
Similar to JAX in Python, users write array programs in plain R, which {anvl} compiles under the hood to optimized machine code using the OpenXLA compiler stack that also underpins frameworks such as TensorFlow and PyTorch/XLA.
It thereby enables the R community to express many numerical algorithms such as neural networks, matrix factorizations, custom optimizers, gradient-based estimators, or Monte Carlo samplers directly in R and run them on both CPUs and GPUs.
A key difference from the existing {torch} package is its reliance on JIT compilation, which can yield significant performance benefits.
A core goal of the package is to offer a low-barrier entry point for R users to leverage GPU acceleration without having to learn a different programming language.
As the package is primarily written in R, it is also easy to extend and contribute to.
Installation with CUDA support is also simplified and only requires a compatible NVIDIA driver to be installed on the system.
As a concrete application, the poster demonstrates how {anvl} can be used to accelerate randomized SVD and thereby enable scaling various bioinformatics workflows to larger data.

Source: <https://github.com/r-xla/anvl>
