# Security, Sandboxing & Operational Safeguards

## 1. Overview
In architectural simulation and automated machine learning experiments, executing hundreds of unattended simulation jobs presents risks of resource exhaustion, infinite loops, runaway processes, and data corruption. This document outlines the security, sandboxing, and resource limits governing the execution environment.

---

## 2. Process Sandboxing & Resource Isolation

### 2.1 Least Privilege Execution
- All gem5 simulations and ML scripts execute under the unprivileged user account (`ashrit`) in WSL (`Ubuntu-24.04`).
- No script or Makefile targets require `sudo` or elevated root access.
- Temporary files and execution artifacts are restricted to the local workspace (`/home/ashrit/gem5/project/outputs/`).

### 2.2 Host Resource Caps (CPU & RAM)
To protect host stability during multi-core simulation sweeps:
- **Concurrency Throttling**: The simulation runner limits concurrent worker processes to $\min(6, \text{logical\_cpus} - 2)$, leaving host cores responsive for system tasks.
- **Memory Limits (`ulimit -v`)**: Each gem5 process is capped at **2.5 GB Virtual RAM**, preventing runaway memory leaks from triggering the Linux OOM killer.
- **Watchdog Timeout Guard**: Every spawned gem5 simulation subprocess is wrapped with a strict 300-second (5 minute) hard timeout. If an execution exceeds 300 seconds, `subprocess.Popen.kill()` is invoked immediately, and the PID tree is reaped.

```python
# Implementation pattern in run_parallel_sweep.py
try:
    stdout, stderr = proc.communicate(timeout=300)
except subprocess.TimeoutExpired:
    proc.kill()
    proc.communicate()
    log_failure(run_id, "TIMEOUT_EXPIRED")
```

---

## 3. Binary & Code Integrity

### 3.1 ELF Binary Validation
Before launching simulations on cross-compiled benchmarks, the framework enforces automated static ELF validation:
- Architecture must match: `AArch64` (`ELF 64-bit LSB executable, ARM aarch64`).
- Linkage must be static: `statically linked`.
- Dynamic linker requests or x86/RISCV binaries are rejected immediately to prevent simulation panics.

### 3.2 Command Injection Prevention
- All CLI parameters passed into gem5 subprocesses are validated through Python's `argparse` type validators.
- Shell invocation (`shell=True`) is strictly prohibited. Subprocesses are executed exclusively using direct argument vectors (`subprocess.Popen(cmd_list, shell=False)`).

---

## 4. Data & Dataset Integrity Protection

### 4.1 SHA-256 Checksumming
- Upon completing dataset collection, `scripts/parse_dataset.py` generates a cryptographic checksum manifest:
  ```
  dataset/dataset_cleaned.csv.sha256
  ```
- Before training, `ml/train_models.py` verifies the hash to guarantee that data was not inadvertently modified or corrupted between phases.

### 4.2 Stat File Immutability
- Raw simulation outputs (`outputs/run_*/stats.txt`) are marked read-only post-simulation (`chmod 444`) to preserve raw provenance for academic auditability.
