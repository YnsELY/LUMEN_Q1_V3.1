# FE8 codec — real, exact entropy coding of doubled-E8 weight codes

Files: `fcodec.py` (API), `fe8c.c` (C range-coder core; compiled with `gcc -O3` the first time it is imported,
cached as `_fe8c_<sha1>.so`), `package.py` (package directory, verify CLI, streaming workers),
`test_fcodec.py` (tests and measurements; output in `test_fcodec.log`).

```python
blob = fcodec.encode_matrix(U, steps_fp16, extra=None)   # U int (m, n/8, 8) or (m, n); every 8-vector must be a doubled-E8 point
U, steps, extra = fcodec.decode_matrix(blob)              # U int32 (m, n); W = U * steps[:, None] / 2 (float32)
```

## Models (one is chosen per matrix from an exact size estimate, model tables included)
Every 8-vector u has coset c = u mod 2 (all coordinates share it) and sum(u) % 4 == 0.
* **mode 0 SHELL**: shell index s = |u|²/8 is coded with a categorical table, then the point inside the shell is coded
  *exactly uniformly*. This is enumerative coding done by the range coder: coset, then u_1..u_8 one after another,
  using exact counts T[c][k][r][s] (the number of k-vectors with coordinates of parity c, squared norm r, and coordinate
  sum ≡ s mod 4). The cost is the ideal `bits_from_hist` cost almost exactly (about +0.005 %). Usable for shells
  0..251 (the count must stay below 2^32); a vector in a larger shell, or with |u_i| > 65535, is coded as an escape
  symbol plus 8×32 raw bits.
* **mode 1 PROD**: the coset bit c, then v_i = (u_i − c) >> 1 for i < 7 with one table per coset. The E8 constraint is
  the same in both cosets: sum(v) is even. So the parity p of v_8 is implied, and only k = v_8 >> 1 is coded, with a
  table per (c, p). This conditioning is essential: without p the model costs +1.6 % at 2 bpw (measured in an
  exploration, in entropy). With p it is +0.01–0.03 % on Gaussian data, and **below** the ideal on heavy-tailed
  coordinates (Student-t: −3.1 %) and on the real weights, because it does not assume a uniform distribution inside
  each shell. A value outside the [-2048, 2048] window is coded as an escape plus 32 raw bits.
* Measured alternatives that were rejected (empirical entropy, real gate_proj at 2.82 bpw, ideal 2.8219): one shared
  iid coordinate table 2.8444 (+0.8 %); per-coset tables without the parity context 2.8290; context on the previous
  coordinate's magnitude or on a running magnitude made no difference (±0.0005). A hybrid that codes small shells
  exactly and large ones with the product model, or the product model conditioned on a coded magnitude bucket, was
  always *worse* (+0.1…+6 %), because the probability mass is double-counted.
* On real weights the selector picks PROD everywhere; SHELL wins on Gaussian synthetic data at ≤ 3.5 bpw. Both are
  always within 0.05 % of each other below about 3.5 bpw, and SHELL degrades above that (shells > 251 escape).

Range coder: 64-bit low/range, byte-wise renormalisation below 2^56, carry propagated in the output buffer. Totals are
≤ 2^32, so each symbol wastes < 2^-24 of its relative precision. Categorical tables are normalised to 2^18. Decoding
uses a 4096-entry lookup table per model followed by a short linear scan.

## Byte layout (little endian)
```
header 32 B : magic "FE8CODC1" | u16 version=1 | u8 mode | u8 P=18 | u32 m | u32 n | u64 payload_len | u32 crc32(payload)
payload     : steps  m × fp16
              extras u16 count, then per item: u8 kind (0 = ndarray, 1 = JSON) | u8 len + name |
                     ndarray: u8 len + numpy dtype.str | u8 ndim | u64 shape[ndim] | u64 nbytes | raw bytes
                     JSON:    u64 len | utf-8 bytes
              tables u8 ntab, then per table: varint zigzag(lo) | varint nsym | freqs (varint; a 0 is followed by a run length)
                     mode 0: [shell table, the last symbol is the escape]
                     mode 1: [coset, v|c=0, v|c=1, k|c0p0, k|c0p1, k|c1p0, k|c1p1], the last symbol of each is the escape
              u64 stream_len | range-coded stream
```
Decoding checks the magic, the version and the crc32 of the payload. Decoded U and steps are bit-exact, and
U*steps/2 is bit-identical to `quantize_rtn`'s Q.

## Measured (test_fcodec.log; single core, Python overhead included)
Real Qwen3.8-27B slices (gate, down, in_proj_qkv, q_proj, o_proj, embed_tokens, lm_head at 2.82 bpw; o_proj and
lm_head also at 4.5 bpw): real/ideal = **0.99996** overall (per matrix 0.9995 … 1.0001), where ideal =
`bits_from_hist` (limit 16384) + 16 bits per row. Encode is about 100–115 M weights/s and decode about 73 M weights/s
per core. Gaussian synthetic data at 2.0 / 2.8 / 3.5 / 4.5 / 6.5 bpw: +0.005 % … +0.03 % (SHELL mode at about
45 / 35 M weights/s). Extrapolated: 26.893e9 weights at 2.808 bpw gives 9.4394 GB ideal and 9.4390 GB real.

## Limits
* Fixed overhead is about 32 B of header plus about 10–100 B of tables per matrix. Matrices with fewer than ~10^4
  weights are dominated by it: 1×8 takes 64 bits. The reference ideal header is 656 B, which is larger.
* |u_i| < 2^31 (int32). Inputs that are not E8 points raise ValueError. A single matrix needs about 4 B per weight of
  RAM for the int32 copy.
* `gcc` is needed on the machine that first imports the module (the .so is then cached in this folder).
* package workers: a file claimed by a worker that crashes stays as `*.npz.claimed.<pid>` and must be renamed back by
  hand. The manifest is updated under an flock with an atomic replace.
