# CPU–GPU Data Transfer Bandwidth using CUDA

This project measures the effective memory transfer bandwidth between host (CPU) memory and device (GPU) memory using CUDA.

It evaluates:

- Host to Device (H2D) bandwidth
- Device to Host (D2H) bandwidth
- The impact of host memory type:
  - Pageable host memory (standard `malloc`)
  - Pinned host memory (`cudaHostAlloc`)

Results are reported as a function of transfer size (1 MB to 1 GB) and visualized as a bandwidth vs. array size plot.

---

## Project Structure

```text
.
├─ main.cu              # Entry point, runs all tests and prints CSV
├─ bandwidth.cu         # Implementation of bandwidth measurements
├─ bandwidth.h          # API and data structures for measurements
├─ cuda_utils.cuh       # CUDA error checking macro (CHECK_CUDA)
├─ Makefile             # Build script using nvcc
├─ plot_bandwidth.py    # Python script to generate bandwidth plot
├─ report.tex           # LaTeX report (optional, for writeup)
├─ bandwidth.csv        # Generated: raw measurement data (CSV)
└─ bandwidth_plot.png   # Generated: bandwidth vs. array size figure
```

---

## Requirements

### Hardware

- NVIDIA GPU with CUDA capability

### Software

- CUDA Toolkit installed (including `nvcc`)
- C++ compiler supported by your CUDA version
- Python 3 (for plotting)
- Python packages:
  - `numpy`
  - `matplotlib`
- LaTeX distribution (for compiling `report.tex`, optional)

Example installation for Python packages:

```bash
pip install numpy matplotlib
```

---

## Building the Project

The project is built using the provided `Makefile`.

From the project root directory:

```bash
make
```

This will:

- Compile `main.cu` and `bandwidth.cu` with `nvcc`
- Produce an executable named `bandwidth`

To clean build artifacts:

```bash
make clean
```

---

## Running the Bandwidth Benchmark

After building, run the benchmark and save the output to a CSV file:

```bash
./bandwidth > bandwidth.csv
```

This command will:

- Measure bandwidth for array sizes from 1 MB to 1 GB (powers of 2)
- Test both directions:
  - Host to Device (H2D)
  - Device to Host (D2H)
- Test both host memory types:
  - Pageable (`malloc`)
  - Pinned (`cudaHostAlloc`)
- Write a CSV file with the following header:

```text
size_MB,h2d_pageable_GBs,d2h_pageable_GBs,h2d_pinned_GBs,d2h_pinned_GBs
```

You can quickly inspect the file using:

```bash
head bandwidth.csv
```

---

## Generating the Bandwidth Plot

Once `bandwidth.csv` has been created, use the Python script to generate a plot:

```bash
python plot_bandwidth.py
```

or, depending on your environment:

```bash
python3 plot_bandwidth.py
```

This script will:

- Load `bandwidth.csv`
- Plot bandwidth (GB/s) vs. transfer size (MB)
- Include four curves:
  - H2D – Pageable
  - D2H – Pageable
  - H2D – Pinned
  - D2H – Pinned
- Save the figure as:

```text
bandwidth_plot.png
```

If you are on a system with a display, a window will also show the plot.

---

## Interpreting the Results

Typical observations:

- For small transfer sizes (e.g., 1–4 MB), measured bandwidth is low and noisy due to fixed overheads dominating (latency bound).
- As transfer size increases (e.g., 8–32 MB), the effective bandwidth rises and approaches a plateau that reflects the maximum practical throughput.
- For large transfer sizes (e.g., ≥ 64 MB):
  - Pinned memory usually achieves higher and more stable bandwidth than pageable memory.
  - H2D and D2H bandwidths may differ slightly due to hardware and driver asymmetries.

In general:

- Pinned memory is recommended for large, performance-critical transfers.
- Grouping many small transfers into fewer large transfers improves overall efficiency.

---

## LaTeX Report

The file `report.tex` contains a complete report template that:

- Explains the background (H2D, D2H, pageable vs. pinned)
- Describes the methodology (CUDA events, array sizes, configuration)
- Includes:
  - A formatted table of the measured results
  - A reference to the generated plot `bandwidth_plot.png`
  - Full source code listings and the plotting script

To compile the report to PDF:

```bash
pdflatex report.tex
pdflatex report.tex
```

The second run ensures that all references (figure, table, etc.) are updated.

---

## Notes and Troubleshooting

- If the executable fails with CUDA errors, check:
  - That the correct GPU device is selected (the code uses device 0 by default).
  - That your GPU supports the CUDA version installed.
  - That the system has enough free GPU memory for the largest array size.
- If very large sizes fail due to memory constraints, you can:
  - Reduce the maximum size in `main.cu`.
  - Decrease the number of repetitions (`num_iters`).

---

## License

You may adapt a license of your choice (for example MIT, BSD, etc.) depending on your use case and course requirements.
