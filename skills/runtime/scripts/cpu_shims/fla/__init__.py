"""CPU-only stand-in for the small subset of flash-linear-attention (fla) that the
OpenArchEvo zoo imports. Pure PyTorch reference implementations; numerically
equivalent in intent (not bit-identical to the Triton kernels). For CPU smoke
checks and demos only, never for reported research metrics."""
__version__ = "cpu-shim-0.1"
