# anvil: JIT compilation, GPU acceleration, and automatic differentiation for R

Due to the interpreted nature of the R language, numerical algorithms are often implemented in a lower-level language such as C, C++ or Fortran.
With {anvil}, you can write numerical programs in R and jit-compile them into optimized machine code using the OpenXLA compiler infrastructure.
Code can run on GPUs and CPUs and we offer support for first-order automatic differentiation.
We will show how to use the package and demonstrate its capabilities using practical examples.
Because the package is primarily implemented in R, it is also easy to extend and contribute to.
The source code for the package is available on GitHub at <https://github.com/r-xla/anvil>.
