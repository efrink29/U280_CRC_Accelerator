# Hash Calculation — Implementation Plan

## Overview

Extend the U280 CRC Accelerator to support **hash computation** (e.g. SHA-256, MD5)
alongside the existing CRC and TCP checksum modes. The goal is to reuse the same
FPGA infrastructure (kernel loading, task queue, manager) so that the user can
switch between CRC, TCP checksum, and hash with a single loaded xclbin — no
reload required.

---

## Current Status (2026-02-10)

- SHA-256 mode is implemented end-to-end (kernel, manager, host reference, and result printing).
- `sw_emu` build and execution are passing for `CRC_32`, `TCP_CHECKSUM`, and `SHA_256`.
- Host-vs-FPGA comparison is passing for tested chunk outputs (`[OK]` across all tested modes).
- Remaining open items are `hw_emu`/hardware runs, formal performance report, and synthesis utilization report capture.

---

## Phase 1: Algorithm Selection & Design Decisions

Phase 1 implementation artifact: `plans/hash_phase1_decisions.md` (completed on 2026-02-10).

### 1.1 Choose Target Hash Algorithm(s)

| Algorithm | Output Size | Block Size | FPGA Complexity | Priority |
|-----------|-------------|------------|-----------------|----------|
| SHA-256   | 256 bits    | 512 bits   | Medium          | **High** |
| SHA-1     | 160 bits    | 512 bits   | Medium          | Medium   |
| MD5       | 128 bits    | 512 bits   | Medium          | Low      |
| SHA-512   | 512 bits    | 1024 bits  | High            | Low      |

**Recommendation:** Start with **SHA-256** — it is the most widely used and has a
good balance of security and FPGA resource usage.

### 1.2 Design Decisions

- [x] Decide on chunk-based hashing (hash per chunk) vs. streaming (hash of full data)
  - Current CRC/TCP architecture uses **chunk-based** processing — hash should follow
    the same pattern for consistency (one hash output per chunk)
- [x] Decide output format: raw 256-bit digest stored as 8 × `uint32_t` per chunk
- [x] Decide whether to support HMAC (keyed hashing) in a future phase
- [x] Evaluate FPGA resource budget on U280 (LUTs, BRAMs, DSPs) to determine how
  many parallel hash compute units can fit alongside existing CRC logic

### 1.3 Deliverables

- [x] Algorithm selection document
- [x] Resource estimation for U280
- [x] Decision on output format and chunk behavior

---

## Phase 2: Extend the Configuration & Mode System

### 2.1 Add New Check Mode

**File:** `src/manager.hpp`

```cpp
enum KernelCheckMode : uint32_t {
    CHECK_MODE_CRC            = 0,
    CHECK_MODE_TCP_CHECKSUM   = 1,
    CHECK_MODE_HASH           = 2   // <-- NEW
};
```

Also mirror in `src/kernel.cpp`:
```cpp
enum KernelCheckMode : uint32_t {
    CHECK_MODE_CRC            = 0,
    CHECK_MODE_TCP_CHECKSUM   = 1,
    CHECK_MODE_HASH           = 2   // <-- NEW
};
```

### 2.2 Extend `KernelConfig`

**File:** `src/manager.hpp`

```cpp
struct KernelConfig {
    uint32_t polynomial;
    uint32_t init_val;
    uint32_t xor_out;
    bool refInput;
    bool refOutput;
    int crcWidth;
    int chunkSize;
    size_t dataSize;
    uint32_t checkMode = 0;
    uint32_t hashAlgorithm = 0;   // <-- NEW: 0=SHA256, 1=SHA1, 2=MD5, etc.
};
```

### 2.3 Extend `loadConfig()` in `host.cpp`

Add a new token parser to recognize hash mode from the config file:

```cpp
if (modeToken == "HASH" || modeToken == "SHA256" || modeToken == "2") {
    cfg.checkMode = CHECK_MODE_HASH;
    cfg.hashAlgorithm = 0; // SHA-256
}
```

### 2.4 Create Example Config File

**File:** `Configs/SHA_256`

```
256
00000000
00000000
0
0
00000000
1024
HASH
```

> Note: polynomial, init_val, xor_out, reflect fields are unused for hash mode
> but kept for config file format consistency. `chunkSize` determines how many
> bytes are hashed per output digest.

### 2.5 Deliverables

- [x] Updated `KernelCheckMode` enum in both `manager.hpp` and `kernel.cpp`
- [x] Updated `KernelConfig` struct with `hashAlgorithm` field
- [x] Updated `loadConfig()` to parse hash mode
- [x] Example config file `Configs/SHA_256`

---

## Phase 3: HLS Kernel Implementation

### 3.1 SHA-256 Constants & Types

**File:** `src/kernel.cpp` (or a new `src/sha256_kernel.cpp` if building a separate kernel)

Define SHA-256 constants in the kernel:

```cpp
// SHA-256 initial hash values (H0..H7)
static const uint32_t SHA256_H[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// SHA-256 round constants (K0..K63)
static const uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    // ... (all 64 values)
};
```

### 3.2 Kernel Functions to Implement

| Function | Purpose |
|----------|---------|
| `sha256_pad_block()` | Pad the last block of a chunk to 512 bits with length encoding |
| `sha256_compress()` | One round of SHA-256 compression (64 rounds per 512-bit block) |
| `sha256_process_chunk()` | Process an entire chunk: iterate over 512-bit blocks, call compress |
| `process_hash()` | Top-level: loop over chunks, call `sha256_process_chunk()`, write results |

### 3.3 Integration into Existing Kernel

**Option A — Single kernel with 3 modes (recommended for simplicity):**

```cpp
extern "C" {
void calculate_crc(const unsigned char *data_in,
                   uint32_t *crc_out,
                   const uint32_t *tables,
                   const unsigned int numChunks,
                   const unsigned int chunkSize,
                   const uint32_t crc_size,
                   const uint32_t init_value,
                   const uint32_t mode)
{
    if (mode == CHECK_MODE_TCP_CHECKSUM) {
        process_tcp_checksum(data_in, crc_out, numChunks, chunkSize);
        return;
    }
    if (mode == CHECK_MODE_HASH) {
        process_hash(data_in, crc_out, numChunks, chunkSize);
        return;
    }
    // ... existing CRC dataflow pipeline
}
}
```

**Option B — Separate kernel binary:**
- Create `src/hash_kernel.cpp` with its own `calculate_hash` entry point
- Requires modifying the xclbin build (Makefile / v++ link step) to include both kernels
- More flexible but increases build complexity

**Recommendation:** Start with **Option A** for faster iteration. Migrate to
Option B later if resource contention becomes an issue.

### 3.4 HLS Optimization Considerations

- SHA-256 compression has 64 sequential rounds per block — pipeline the
  message schedule expansion (`W[t]` computation) to overlap with compression
- Use `#pragma HLS PIPELINE II=1` on the inner compression loop where possible
- The 512-bit input block can be read as 16 × 32-bit words from the existing
  byte stream infrastructure
- Output is 8 × `uint32_t` per chunk (256 bits) — the `crc_out` buffer must be
  sized accordingly: `numChunks * 8` words instead of `numChunks * 1`

### 3.5 Output Buffer Sizing

**Important:** CRC/TCP produce 1 × `uint32_t` per chunk.
SHA-256 produces **8 × `uint32_t`** per chunk.

The output buffer must be resized:
```cpp
// In kernel.cpp — process_hash
for (unsigned int c = 0; c < numChunks; ++c) {
    uint32_t digest[8];
    sha256_process_chunk(data_in + c * chunkSize, chunkSize, digest);
    for (int i = 0; i < 8; ++i)
        crc_out[c * 8 + i] = digest[i];
}
```

### 3.6 Deliverables

- [x] SHA-256 constants defined in kernel source
- [x] `sha256_compress()` HLS function with proper pragmas
- [x] `sha256_process_chunk()` for full chunk hashing with padding
- [x] `process_hash()` top-level loop integrated into `calculate_crc`
- [ ] HLS synthesis report showing resource usage and latency

---

## Phase 4: Host-Side Reference Implementation

Phase 4 implementation status: completed and integrated into the host verification path.

### 4.1 Software SHA-256 for Verification

**File:** `src/helpers/sha256.h` and `src/helpers/sha256.cpp`

Implement a software SHA-256 to serve as the **golden reference** for verifying
FPGA results. This mirrors the existing pattern where `fast_crc()` in `host.cpp`
provides a host-side CRC reference.

```cpp
// sha256.h
#ifndef SHA256_H
#define SHA256_H

#include <cstdint>
#include <cstddef>
#include <vector>

// Compute SHA-256 digest of `data[0..len-1]`
// Returns 8-element vector of uint32_t (256 bits)
std::vector<uint32_t> sha256(const unsigned char *data, size_t len);

#endif
```

### 4.2 Extend `fast_crc()` in `host.cpp`

Add a hash branch to the host-side reference function:

```cpp
std::vector<uint32_t> fast_crc(const Bytes &data, const KernelConfig &cfg) {
    size_t chunkBytes = static_cast<size_t>(cfg.chunkSize);
    size_t nChunks = data.size() / chunkBytes;

    if (cfg.checkMode == CHECK_MODE_TCP_CHECKSUM) {
        // ... existing TCP checksum code ...
    }

    if (cfg.checkMode == CHECK_MODE_HASH) {
        std::vector<uint32_t> results;
        results.reserve(nChunks * 8);
        for (size_t chunk = 0; chunk < nChunks; ++chunk) {
            auto digest = sha256(data.data() + chunk * chunkBytes, chunkBytes);
            results.insert(results.end(), digest.begin(), digest.end());
        }
        return results;
    }

    // ... existing CRC code ...
}
```

### 4.3 Deliverables

- [x] `src/helpers/sha256.h` — header
- [x] `src/helpers/sha256.cpp` — software SHA-256 implementation
- [x] Updated `fast_crc()` with hash branch
- [x] Include `sha256.h` in `host.cpp`

---

## Phase 5: Manager & FPGA Pipeline Updates

### 5.1 Update `execute_crc()` in `manager.hpp`

Handle the new mode in the FPGA execution path:

```cpp
static std::vector<uint32_t> execute_crc(Worker &w,
                                         const std::vector<unsigned char> &data,
                                         const KernelConfig &cfg)
{
    // ... existing setup ...

    // For hash mode: output is 8 words per chunk, not 1
    size_t wordsPerChunk = (cfg.checkMode == CHECK_MODE_HASH) ? 8 : 1;

    // ... set kernel args (mode is already passed as arg 7) ...

    // Adjust output buffer read size
    std::vector<uint32_t> crcOut(chunksToProcess * wordsPerChunk);
    err = w.qD2H.enqueueReadBuffer(
        w.dOutA, CL_TRUE, 0,
        sizeof(uint32_t) * chunksToProcess * wordsPerChunk,
        crcOut.data(), &waitList2, &evD2H);

    // Skip CRC-specific post-processing for hash mode
    if (cfg.checkMode == CHECK_MODE_CRC) {
        // existing XOR-out and reflect logic
    }
    // Hash results are returned as-is (no post-processing needed)

    return result;
}
```

### 5.2 Update Buffer Sizing

The output buffer `dOutA` must be large enough for hash output:

```cpp
// In init_worker() — manager.hpp
// Current: w.dOutA = cl::Buffer(..., buffer_size, ...);
// This already allocates buffer_size bytes which should be sufficient
// as long as buffer_size >= numChunks * 8 * sizeof(uint32_t)
```

Verify that `buffer_size` is adequate. For hash mode with chunk size `C`:
- Chunks per buffer = `buffer_size / C`
- Output words = `chunks_per_buffer * 8`
- Output bytes = `chunks_per_buffer * 32`

This must be ≤ `buffer_size`. As long as `C >= 32` bytes (which it always is for
meaningful hashing), this constraint is satisfied.

### 5.3 Table Handling

For hash mode, CRC tables are not needed. The existing logic already handles this:
- When `checkMode != CHECK_MODE_CRC`, zeroed tables are uploaded (manager.hpp line 473)
- The kernel ignores tables when `mode == CHECK_MODE_HASH`

No changes needed here.

### 5.4 Deliverables

- [x] Updated `execute_crc()` with hash-aware output sizing
- [x] Verified buffer sizing for hash output
- [x] No table upload changes needed

---

## Phase 6: Build System Updates

### 6.1 Update Makefile / Build Scripts

- [x] Add `src/helpers/sha256.cpp` to the host compilation targets
- [x] If using Option B (separate kernel), add the new kernel to v++ compile and
  link steps
  - N/A for current implementation because Option A (single kernel with mode switch) is used
- [x] Update the xclbin build to include the hash logic in the kernel

### 6.2 Connectivity Configuration

If using a separate kernel, update the `connectivity.cfg` (or equivalent):

```ini
[connectivity]
nk=calculate_crc:10:CRC_0.CRC_1.CRC_2.CRC_3.CRC_4.CRC_5.CRC_6.CRC_7.CRC_8.CRC_9
```

No changes needed if hash is integrated into the existing `calculate_crc` kernel
(Option A).

### 6.3 Deliverables

- [x] Updated Makefile / build scripts
- [x] SW emulation (`sw_emu`) build passing
- [ ] HW emulation (`hw_emu`) build passing
- [ ] Hardware build initiated (if applicable)

---

## Phase 7: Testing & Verification

### 7.1 Software Emulation (`sw_emu`)

- [x] Run with `Configs/SHA_256` config
- [x] Compare FPGA results against host-side `sha256()` reference
- [x] Test with known test vectors (e.g. NIST SHA-256 test vectors):
  - Empty string: `e3b0c44298fc1c149afbf4c8996fb924...`
  - `"abc"`: `ba7816bf8f01cfea414140de5dae2223...`
  - 1 million × `"a"`: `cdc76e5c9914fb9281a1c7e284d73e67...`

### 7.2 Functional Tests

- [x] Chunk-based hashing produces correct digest per chunk
- [ ] Mode switching: CRC → Hash → TCP → CRC without kernel reload
- [ ] Multiple compute units running hash in parallel
- [ ] Edge cases: chunk size equals block size (64 bytes), chunk with exact
  block alignment, chunk requiring padding

### 7.3 Performance Benchmarking

- [ ] Measure hash throughput (MB/s) for various chunk sizes
- [ ] Compare FPGA hash throughput vs. host-side software SHA-256
- [ ] Compare against existing CRC/TCP throughput for context
- [ ] Profile resource usage on U280

### 7.4 Deliverables

- [x] sw_emu test results
- [ ] hw_emu test results
- [x] Known-answer test (KAT) results
- [ ] Performance report (throughput, latency, FPGA utilization)

Latest sw_emu validation snapshot (2026-02-10):
- Command: `printf "4\n\n" | plans/run_swemu_compare.sh`
- Input: `BinaryData/hash_10mb.bin` (10 MB)
- Modes tested: `CRC_32`, `TCP_CHECKSUM`, `SHA_256`
- Result: host and FPGA outputs matched (`[OK]`) for all reported sample outputs

---

## Phase 8: Result Display & Output

### 8.1 Update `print_result()` in `host.cpp`

Hash output is 256 bits (8 × `uint32_t`), not a single 32-bit value.
Add a hash-specific print function:

```cpp
void print_hash_result(const char *label, const uint32_t *result,
                       const uint32_t *expected, size_t numWords) {
    std::cout << label;
    for (size_t i = 0; i < numWords; ++i)
        print_hex_value(result[i], 32);
    // Compare
    bool match = std::memcmp(result, expected, numWords * sizeof(uint32_t)) == 0;
    std::cout << (match ? " [OK]" : " [MISMATCH]") << std::endl;
}
```

### 8.2 Update `main()` Result Comparison Loop

```cpp
if (crc32.checkMode == CHECK_MODE_HASH) {
    for (size_t i = 0; i < std::min(resHost.size() / 8, (size_t)5); i++) {
        print_hash_result("Host: ", &resHost[i * 8], &resHost[i * 8], 8);
        print_hash_result("FPGA: ", &resFPGA[i * 8], &resHost[i * 8], 8);
    }
} else {
    // existing CRC/TCP result printing
}
```

### 8.3 Deliverables

- [x] Hash-specific result printing functions
- [x] Updated main() with hash result display

---

## File Change Summary

| File | Changes |
|------|---------|
| `src/manager.hpp` | Add `CHECK_MODE_HASH`, extend `KernelConfig`, update `execute_crc()` output sizing |
| `src/kernel.cpp` | Add `CHECK_MODE_HASH`, implement `process_hash()` with SHA-256 |
| `src/host.cpp` | Update `loadConfig()`, `fast_crc()`, result printing in `main()` |
| `src/helpers/sha256.h` | **NEW** — SHA-256 header |
| `src/helpers/sha256.cpp` | **NEW** — SHA-256 software reference implementation |
| `src/helpers/crc.h` | No changes needed |
| `Configs/SHA_256` | **NEW** — Example hash config file |
| `Makefile` | Add sha256.cpp to build |

---

## Risk & Considerations

1. **FPGA Resource Usage**: SHA-256 has 64 sequential rounds per block. Achieving
   high throughput requires careful pipelining. The 16-byte parallel stream
   architecture used for CRC may need adaptation since SHA-256 operates on
   64-byte (512-bit) blocks.

2. **Output Buffer Mismatch**: The biggest integration risk is the output size
   difference (1 word for CRC/TCP vs. 8 words for SHA-256). This must be
   consistently handled in the kernel, manager, and host code.

3. **Kernel Size**: Adding SHA-256 logic to the existing `calculate_crc` kernel
   will increase its FPGA footprint. If resource contention occurs, consider
   splitting into a separate kernel (Option B from Phase 3).

4. **Build Time**: HW builds on U280 take several hours. Plan to validate
   thoroughly in sw_emu and hw_emu before starting a hardware build.

5. **Future Extensibility**: The mode-based architecture scales well. Adding
   more hash algorithms later (SHA-1, MD5, SHA-512) follows the same pattern —
   add a new mode value and implement `process_<algorithm>()` in the kernel.
