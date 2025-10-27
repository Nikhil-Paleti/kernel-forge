# GPU Kernel Library

> **A growing collection of custom GPU kernels for core machine learning and deep learning operations — implemented in Triton, CUDA, and Metal.**  
> Built for learning, benchmarking, and optimizing GPU programming across frameworks.

---

## Overview

This project explores how fundamental **machine learning operators** — such as matrix multiplication, softmax, convolutions, and activation functions — are implemented and optimized at the GPU level.

Each kernel is written **from scratch**, beginning in **Triton**, then extended to **CUDA** and **Metal**.

---

## Currently Working On
- Implementing kernels in **Triton** for correctness and clarity.  

## Future Plans
- Port all kernels to **CUDA** and **Metal**.  
- Add optimization guides, roofline plots, and Nsight profiling notes.  

---

## Kernels Implemented

### Linear Algebra
- **Matrix Multiplication (GEMM)** – base operator behind most ML models.  
- **Matrix Transpose** – transpose a matrix.  
- **Matrix Copy** – copy a matrix using gpu.

### Convolution & Filtering
- **1D Convolution** – foundation for conv1d / feature extraction kernels.

### Activation Functions
- **ReLU** – standard nonlinear activation.  
- **Leaky ReLU** – variant with negative slope.  
- **Sigmoid Linear Unit (SiLU / Swish)** – smooth activation for transformers.  
- **Swish-Gated Linear Unit (SwiGLU)** – used in large language models (LLMs).  
- **Softmax** – numerically stable reduction, core of attention mechanisms.

### Elementwise & Array Operations
- **Vector Addition** – baseline memory bandwidth kernel.  
- **Color Inversion** – image pixel operation example.  
- **Reverse Array** – strided memory access pattern.

### Reductions & Counting
- **Count Array Elements** – simple reduction / histogram base.  
- **Count 2D Array Elements** – same as above.

### Miscellaneous
- **Rainbow Table** – demonstration kernel for hashing / lookup patterns.

---