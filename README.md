# Accelerating LLM Inference with Multi-Threading
MULTITHREADED PROGRAMMING AND SYNCHRONIZATION

**By Zhao Wenzhe (UID: 3035xxxxxx)**

---

## Overview
This project implements a multi-threaded inference engine for the Llama3 language model, parallelizing **matrix-vector multiplication** and **multi-head attention** computations using POSIX Pthreads. Key features:
- Thread pool design to reuse threads and reduce overhead
- Synchronization via mutex locks and condition variables
- Benchmarking framework for performance analysis

---

## Environment Setup

### 1. Download Model Files
The model files (`model.bin` and `tokenizer.bin`) are **not included in the repository**. Run:
```bash
make prepare  # Automatically downloads required files
Manual download (not recommended):

wget -O model.bin https://huggingface.co/huangs0/smollm/resolve/main/model.bin
wget -O tokenizer.bin https://huggingface.co/huangs0/smollm/resolve/main/tokenizer.bin
2. Compilation
Sequential Version (Baseline):

make -B seq  # Force rebuild seq.c
Parallel Version (Multi-Threaded):

make -B      # Compiles parallel_3035844883.c
Requirements:

GCC ≥9.4.0 (check with gcc -v)

Linux environment (tested on workbench2.cs.hku.hk)

Usage
Run Inference

# Sequential Version
./seq <seed> "<prompt>"
# Example:
./seq 42 "What's Fibonacci Number?"

# Parallel Version
./parallel <thread_count> <seed> "<prompt>"
# Example (4 threads):
./parallel 4 42 "Why didn't my parents invite me to their wedding?"
Output Format
User: [Your Prompt]
assistant: [Generated Response]
length: [Token Count], speed (tok/s): [Throughput]
[Thread System Usage]
main thread - user: [Time] s, system: [Time] s
whole process - user: [Time] s, system: [Time] s
Performance Report
Benchmark results on local machine (see report_3035.pdf for details):

Threads	Speed (tok/s)	User Time	System Time	User/System Ratio
0 (Seq)	23.32	11.43s	0.17s	67.28
4	41.30	12.07s	3.54s	3.41
8	36.24	12.79s	5.84s	2.19
Key Findings:

Optimal Threads: 4 threads achieve maximum speedup (41.3 tok/s, +77% vs sequential)

Overhead Trade-off: Beyond 4 threads, system time increases significantly due to synchronization costs

File Structure
.
├── parallel_3035844883.c    # Multi-threaded implementation (main submission)
├── seq.c                    # Baseline sequential version
├── common.h                 # Helper macros & utilities
├── model.h                  # Model definition (DO NOT MODIFY)
├── Makefile                 # Build script (update UID if reusing)
├── report_3035.pdf          # Performance analysis report
└── (model.bin/tokenizer.bin # Auto-downloaded by `make prepare`)
Submission Notes
Code Requirements:

Compiles with gcc -O2 -lm -lpthread

No external libraries (e.g., OpenMP/BLAS) allowed

Plagiarism Warning:
Source code similarity checks will be performed.

Acknowledgements
Base implementation adapted from llama2.c (Andrej Karpathy)

Model weights from SmollM (HuggingfaceTB)


---

### Appendix: Example Output
```text
$ ./parallel 4 42 "What is Fibonacci Number?"
User: What is Fibonacci Number?
assistant: A Fibonacci sequence is a sequence of numbers...
length: 266, speed (tok/s): 38.8889
Thread 0 - user: 4.9396 s, system: 0.1620 s
Thread 1 - user: 4.7195 s, system: 0.1806 s
Thread 2 - user: 4.6274 s, system: 0.1843 s
Thread 3 - user: 5.0763 s, system: 0.1702 s
main thread - user: 0.6361 s, system: 0.6993 s
whole process - user: 20.0198 s, system: 1.3757 s