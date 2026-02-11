# Hash Phase 1 Decisions (Implemented)

Date: 2026-02-10
Scope: `plans/hash.md` Phase 1

## 1) Algorithm Selection

- Primary algorithm selected: `SHA-256`
- Rationale:
  - Widely used and security-relevant.
  - 512-bit block size aligns well with chunk-based batching.
  - Significantly lower implementation cost than SHA-512 for first integration.

## 2) Chunk Behavior Decision

- Selected behavior: `chunk-based hashing`
- Definition:
  - Each input chunk is hashed independently.
  - One digest is emitted per chunk.
- Why this was selected:
  - Matches existing CRC/TCP per-chunk execution model.
  - Preserves current task splitting and CU scheduling behavior in `manager.hpp`.
  - Keeps mode switching simple (CRC/TCP/HASH on same host flow).

## 3) Output Format Decision

- Digest representation: `8 x uint32_t` per chunk (`256 bits total`).
- Word order: SHA-256 state order `H0..H7` (big-endian digest words).
- Buffer rule:
  - CRC/TCP modes: `1` output word per chunk.
  - HASH mode: `8` output words per chunk.

## 4) HMAC Scope Decision

- Decision: `HMAC is out of scope for current implementation`.
- Plan:
  - Keep a future extension point via `hashAlgorithm` / mode-specific config fields.
  - Revisit once base SHA-256 path is validated in `sw_emu` and `hw_emu`.

## 5) U280 Resource Estimation (Planning-Level)

This estimate is intentionally conservative and used for planning only.

Per SHA-256 CU estimate (typical HLS range):
- LUTs: `20k - 45k`
- FFs: `25k - 60k`
- BRAM(36K): `20 - 60`
- DSP: `0 - 64`

Projected multi-CU budget impact:
- `4 CUs`: low risk, should fit comfortably with current CRC/TCP infrastructure.
- `8 CUs`: moderate risk, expected to fit but routing/timing pressure increases.
- `10+ CUs`: high risk without optimization or reducing parallelism elsewhere.

Initial implementation target:
- Start with `4` hash-enabled CUs in emulation/bring-up.
- Scale upward after synthesis/timing reports confirm margins.

## 6) Phase 1 Deliverables Status

- [x] Algorithm selection document
- [x] Resource estimation for U280
- [x] Decision on output format and chunk behavior
