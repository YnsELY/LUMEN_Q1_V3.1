"""Independent decoder supplied in the user counter-audit, retained with attribution.
Only imports and location-independent extraction changed; no lumen_ref dependency.
"""
import struct, math, hashlib
import numpy as np

IND_HDR = struct.Struct("<8sBBBBIQQQQ")
IND_MAT_HDR = struct.Struct("<8sIIQ")
IND_RANS_L = 1 << 23


def _build_independent_2e8_dp(max_shell: int = 128) -> np.ndarray:
    max_e = 8 * max_shell
    dp = np.zeros((9, max_e + 1, 4, 2), dtype=np.int64)
    dp[0, 0, 0, 0] = 1
    dp[0, 0, 0, 1] = 1
    for k in range(1, 9):
        for p in (0, 1):
            for e in range(max_e + 1):
                lim = math.isqrt(e)
                for a in range(-lim, lim + 1):
                    if (a & 1) != p:
                        continue
                    rem_e = e - a * a
                    for res in range(4):
                        dp[k, e, res, p] += dp[k - 1, rem_e, (res - a) % 4, p]
    return dp


IND_DP = _build_independent_2e8_dp(128)


def independent_unrank_2e8(m: int, r: int, parity: int | None) -> list[int]:
    c0 = int(IND_DP[8, 8 * m, 0, 0])
    c1 = int(IND_DP[8, 8 * m, 0, 1])
    if parity is None:
        if not (0 <= r < c0 + c1):
            raise ValueError("Independent unrank: rank out of range")
        if r < c0:
            p = 0
        else:
            p = 1
            r -= c0
    else:
        p = parity
        lim_c = c0 if p == 0 else c1
        if not (0 <= r < lim_c):
            raise ValueError("Independent unrank: coset rank out of range")

    coords = []
    e = 8 * m
    res = 0
    for idx in range(8):
        rem_k = 7 - idx
        lim = math.isqrt(e)
        found = False
        for a in range(-lim, lim + 1):
            if (a & 1) != p:
                continue
            cnt = int(IND_DP[rem_k, e - a * a, (res - a) % 4, p])
            if r < cnt:
                coords.append(a)
                e -= a * a
                res = (res - a) % 4
                found = True
                break
            r -= cnt
        if not found:
            raise ValueError("Independent unrank failed")
    if e != 0 or res != 0:
        raise ValueError("Independent unrank invariant broken")
    return coords


def independent_decode_le8cpu3(blob: bytes) -> tuple[np.ndarray, dict]:
    if len(blob) < IND_HDR.size + 32:
        raise ValueError("Truncated LE8CPU3")
    body, digest = blob[:-32], blob[-32:]
    if hashlib.sha256(body).digest() != digest:
        raise ValueError("Bad SHA-256")
    magic, mode, limit, prec, rsv, group, n_vec, la, lb, le = IND_HDR.unpack_from(body, 0)
    if magic != b"LE8CPU3\x00" or mode not in (0, 1) or not (1 <= limit <= 128) or not (1 <= prec <= 16) or rsv != 0:
        raise ValueError("Bad LE8CPU3 header")
    if not (1 <= group <= 4096):
        raise ValueError("Bad group_size")
    nf = limit * (2 if mode else 1) + 1
    if nf > (1 << prec):
        raise ValueError("Alphabet exceeds 2^precision")
    off = IND_HDR.size
    if off + 4 * nf + la + lb + le != len(body) or (le % 32) != 0:
        raise ValueError("Section length mismatch")
    freqs = struct.unpack_from(f"<{nf}I", body, off)
    off += 4 * nf
    if any(f <= 0 for f in freqs) or sum(freqs) != (1 << prec):
        raise ValueError("Bad frequency table")

    cdf = [0]
    slot_to_sym = [0] * (1 << prec)
    for sym_i, f in enumerate(freqs):
        c_start = cdf[-1]
        for s_pos in range(c_start, c_start + f):
            slot_to_sym[s_pos] = sym_i
        cdf.append(c_start + f)

    rans_buf = body[off:off + la]
    off += la
    rank_buf = body[off:off + lb]
    off += lb
    esc_buf = body[off:off + le]

    if len(rans_buf) < 4:
        raise ValueError("Truncated rANS state")
    state = struct.unpack_from("<I", rans_buf, 0)[0]
    if not (IND_RANS_L <= state < (IND_RANS_L << 8)):
        raise ValueError("Non-canonical initial state")
    r_pos = 4
    mask = (1 << prec) - 1
    syms = []
    for _ in range(n_vec):
        slot = state & mask
        s = slot_to_sym[slot]
        state = freqs[s] * (state >> prec) + (slot - cdf[s])
        while state < IND_RANS_L:
            if r_pos >= len(rans_buf):
                raise ValueError("Truncated rANS bytes")
            state = (state << 8) | rans_buf[r_pos]
            r_pos += 1
        syms.append(s)
    if state != IND_RANS_L or r_pos != len(rans_buf):
        raise ValueError("Non-canonical rANS end")

    esc_sym = nf - 1
    radices = []
    for s in syms:
        if s == esc_sym:
            continue
        if mode == 0:
            rad = int(IND_DP[8, 8 * s, 0, 0] + IND_DP[8, 8 * s, 0, 1])
        else:
            rad = int(IND_DP[8, 8 * (s // 2), 0, s & 1])
        if rad <= 0:
            raise ValueError("Zero-size coset decoded")
        radices.append(rad)

    def _unpack_tree(val: int, rads: list[int]) -> list[int]:
        if len(rads) == 1:
            return [val]
        mid = len(rads) // 2
        right_prod = math.prod(rads[mid:])
        q_part, r_part = divmod(val, right_prod)
        return _unpack_tree(q_part, rads[:mid]) + _unpack_tree(r_part, rads[mid:])

    ranks = []
    rk_pos = 0
    for g0 in range(0, len(radices), group):
        sub_rad = radices[g0:g0 + group]
        prod = math.prod(sub_rad)
        nb = ((prod - 1).bit_length() + 7) // 8
        if rk_pos + nb > len(rank_buf):
            raise ValueError("Truncated rank word")
        val = int.from_bytes(rank_buf[rk_pos:rk_pos + nb], "big")
        rk_pos += nb
        if val >= prod:
            raise ValueError("Mixed-radix overflow")
        ranks.extend(_unpack_tree(val, sub_rad))
    if rk_pos != len(rank_buf):
        raise ValueError("Trailing rank bytes")

    rk_iter = iter(ranks)
    esc_pos = 0
    out = []
    for s in syms:
        if s == esc_sym:
            if esc_pos + 32 > len(esc_buf):
                raise ValueError("Truncated escape")
            u = list(struct.unpack_from("<8i", esc_buf, esc_pos))
            esc_pos += 32
            if any((c & 1) != (u[0] & 1) for c in u) or (sum(u) % 4) != 0:
                raise ValueError("Invalid E8 escape")
            if (sum(c * c for c in u) // 8) < limit:
                raise ValueError("Non-canonical escape")
            out.append(u)
        else:
            m = (s // 2) if mode else s
            p = (s & 1) if mode else None
            out.append(independent_unrank_2e8(m, next(rk_iter), p))
    if esc_pos != len(esc_buf):
        raise ValueError("Trailing escape bytes")
    arr = np.array(out, dtype=np.int64).reshape(n_vec, 8)
    return arr, {"mode": mode, "limit": limit, "precision": prec, "group": group}


def independent_decode_lmatv3(blob: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(blob) < IND_MAT_HDR.size + 32:
        raise ValueError("Truncated LMATv3")
    body, digest = blob[:-32], blob[-32:]
    if hashlib.sha256(body).digest() != digest:
        raise ValueError("Bad matrix SHA-256")
    magic, m, n, plen = IND_MAT_HDR.unpack_from(body, 0)
    if magic != b"LMATv3\x00\x00" or m == 0 or n == 0 or (n % 8) != 0:
        raise ValueError("Bad LMATv3 dims")
    off = IND_MAT_HDR.size
    if off + 2 * m + plen != len(body):
        raise ValueError("Bad LMATv3 length")
    scales = np.frombuffer(body, dtype="<f2", count=m, offset=off).astype(np.float64)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Bad LMATv3 scales")
    u_flat, _ = independent_decode_le8cpu3(body[off + 2 * m:])
    codes = u_flat.reshape(m, n // 8, 8)
    weights = (codes.reshape(m, n) / 2.0) * scales[:, None]
    return weights, codes, scales


