#!/usr/bin/env python3
"""
Suite Indépendante de Contre-Vérification, Décodage Ex Nihilo et Fuzzing Adversarial — LUMEN Q1-v3-ref
Cible auditée (en LECTURE STRICTEMENT SEULE) : LUMEN_Q1_V3_CORRIGE

Exécution portable depuis n'importe quel répertoire courant :
    python Livrables/contre-verification-v3/test_independent_audit_v3.py

Couvre de façon indépendante, déterministe et reproductible (NumPy 2.3.5) les Sections 2, 3 et 4 du mandat :
  - Test 01 : Intégrité stricte de LUMEN_Q1_V3_CORRIGE et preuve de l'échec initial de verify_manifest.py
              (PROMPT_REVERIFICATION_V3.md absent : 63/64 entrées présentes).
  - Test 02 : Lecteur/décodeur 100 % indépendant de LE8CPU3 et LMATv3 (zéro appel à lumen_ref.entropy ni e8)
              validant les 6 flux results/codec_*.bin, les 240 racines, les grandes coquilles, les 2 co-ensembles,
              ESC, int32 min (-2^31), les flux vides, P=1 (0 octet de rang) et les précisions extrêmes (1 et 16).
  - Test 03 : Fuzzing adversarial avec SHA-256 externe recalculé + reproduction des défauts codec/E8 :
              asymétrie m=0/n=0 de encode_matrix/decode_matrix, troncature silencieuse FP16 (+98,68 % d'erreur sur
              pas subnormal), asymétrie ±1e8 de e8.nearest, TypeError de e8.valid sur bigint, booléens acceptés.
  - Test 04 : Conditions exactes brisant l'invariance de longueur (bascule de co-ensemble à coquille et fréquence
              égales en mode 1 : 725 vs 728 o ; permutation adjacente ESC / non-ESC : 135 vs 134 o ; permutation
              inter-groupes : 348 vs 347 o) et mesure du coût quadratique O(group^2) de _unpack_ranks.
  - Test 05 : Vérifications indépendantes d'Algèbre & LDLQ jusqu'à cond = 1e12 + équivalence exacte de quantize_ldlq
              contre la factorisation dense explicite du produit de Kronecker G (x) H + mise en évidence du terme
              intra-bloc parasite si l'on applique (U_i - I) à la matrice d'erreur finale E_full.
  - Test 06 : Quatre contre-exemples reproductibles issus de l'absence de symétrisation dans checked_spd :
              crash LinAlgError dans upper_ldl, crash ArithmeticError dans refine_quadratic, hausse silencieuse non
              monotone de quadratic_objective, et transposée dans fresh_quadratic_target.
  - Test 07 : Dérivation vectorisée ex nihilo de BGCC (plein rang, portes nulles, Xq de rang 1, down non orthogonal)
              + contre-exemples reproductibles sur pcg (sortie prématurée converged=False à k=20 < maxiter=200,
              faux rejet ValueError "not numerically SPD" par underflow) + gardes NaN manquantes (ridge, trust_radius).
  - Test 08 : RDOQ & polissage : vérification de l'incumbent, de l'alternance des balayages, du bug de métadonnée
              table_refitted_after_sweep=True sous preserve_shell=True, et contre-exemples constructifs où
              refine_quadratic (491 -> 492 o) et refine_swiglu_up (493 -> 501 o) baissent l'objectif à table fixe
              mais AUGMENTENT physiquement la taille encodée après ShellModel.fit.
  - Test 09 : Contrôle exhaustif par énumération brute de allocate_options (20 instances aléatoires faisables et
              infaisables + dépassement max_states) + contre-exemples sur continuous_waterfill (dépassement du budget
              dans [n@lo - 1e-5, n@lo), overflow float64 de n @ hi), booléen accepté comme 1 octet, np.int64 rejeté,
              ceiling_bytes=NaN, et audit arithmétique de budget_lumen_max_v3.py (auxiliaires Mini_text, alignement).
  - Test 10 : Contre-exemples critiques sur metrics.py : indices Top-K dupliqués produisant une KL strictement
              négative (-0,455 nats sur grouped_kl_from_cache et exact_grouped_kl), underflow catastrophique de
              exact_grouped_kl au-delà de -745 (> 54,6 nats d'erreur), overflow FP16 auto-infligé dans topk_cache,
              et inversion de classement à ~3,9e-7 nats due à la quantification FP16 du cache sur V=248320.
  - Test 11 : Contre-exemples sur accept_decoded (plafond NaN désactivant le contrôle d'octets, candidat vide b""
              accepté, nouvelles sentinelles NaN ignorées, rejet d'un candidat plus compact à score égal) et sur
              validate_capture.py (matrices à 0 token (5120, 0), splits vides [] et placeholders acceptés).
  - Test 12 : Audit quantitatif des résultats publiés V3 : non-égalité des octets dans experiences_synthetiques.json
              (+79 octets sur graine 601 entre asymmetric et Qplus), omission du temps CG/ridge hors search_seconds_cpu,
              et vérification finale qu'aucun fichier de LUMEN_Q1_V3_CORRIGE n'a été créé, supprimé ou modifié.
"""

from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import hashlib
import itertools
import json
import math
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[1]
TARGET_V3_ROOT = WORKSPACE_ROOT / "LUMEN_Q1_V3_CORRIGE"
if str(TARGET_V3_ROOT) not in sys.path:
    sys.path.insert(0, str(TARGET_V3_ROOT))
if str(TARGET_V3_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(TARGET_V3_ROOT / "tools"))

from lumen_ref import e8, entropy, linalg, quantization, calibration, budget, metrics
import budget_lumen_max_v3
import validate_capture


def _snapshot_tree(root: Path) -> dict[str, tuple[int, str]]:
    snap = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            rel = p.relative_to(root).as_posix()
            snap[rel] = (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
    return snap


INITIAL_SNAPSHOT = _snapshot_tree(TARGET_V3_ROOT)


# ============================================================================
# A. LECTEUR / DÉCODEUR 100 % INDÉPENDANT DE LE8CPU3 ET LMATv3
# ============================================================================
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


def _make_spd_with_cond(n: int, cond: float, seed_shift: int) -> np.ndarray:
    idx = np.arange(1, n + 1, dtype=np.float64)
    q = np.sin(np.outer(idx, idx + seed_shift) * 1.23456789) + 2.0 * np.eye(n)
    for i in range(n):
        for k in range(i):
            q[i] -= np.dot(q[i], q[k]) * q[k]
        q[i] /= np.linalg.norm(q[i])
    eigs = cond ** (np.arange(n, dtype=np.float64) / max(1, n - 1))
    a = (q.T * eigs[None, :]) @ q
    return 0.5 * (a + a.T)


# ============================================================================
# B. TESTS INDÉPENDANTS ET CONTRE-EXEMPLES REPRODUCTIBLES
# ============================================================================
class IndependentAuditV3Suite(unittest.TestCase):
    def test_01_manifest_missing_file_detected(self):
        manifest = json.loads((TARGET_V3_ROOT / "MANIFEST_SHA256.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["files"]), 64)
        missing = [e["path"] for e in manifest["files"] if not (TARGET_V3_ROOT / e["path"]).is_file()]
        self.assertEqual(missing, ["PROMPT_REVERIFICATION_V3.md"])
        for e in manifest["files"]:
            if e["path"] == "PROMPT_REVERIFICATION_V3.md":
                continue
            p = TARGET_V3_ROOT / e["path"]
            self.assertEqual(p.stat().st_size, e["bytes"])
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), e["sha256"])

    def test_02_independent_decoder_on_saved_and_extreme_streams(self):
        rng = np.random.default_rng(20261112)
        expected = {
            "uniform_shell1": e8.ROOTS[rng.integers(240, size=8192)],
            "integer_coset_shell1": e8.ROOTS[rng.integers(112, size=8192)],
            "gaussian_E8": np.array([e8.nearest(v) for v in rng.normal(size=(8192, 8))]),
        }
        for label, src in expected.items():
            for mode in (0, 1):
                raw = (TARGET_V3_ROOT / "results" / f"codec_{label}_mode{mode}.bin").read_bytes()
                dec, meta = independent_decode_le8cpu3(raw)
                self.assertEqual(meta["mode"], mode)
                np.testing.assert_array_equal(dec, src)

        dec_empty, _ = independent_decode_le8cpu3(entropy.encode_e8([]))
        self.assertEqual(dec_empty.shape, (0, 8))

        extreme_codes = list(e8.ROOTS) + [
            [0] * 8,
            [16, 16, 0, 0, 0, 0, 0, 0],
            [40, 20, 12, 4, 0, 0, 0, 0],
            [-2**31, 0, 0, 0, 0, 0, 0, 0],
        ]
        for mode in (0, 1):
            m_fit = entropy.ShellModel.fit(extreme_codes, limit=64, mode=mode)
            blob = entropy.encode_e8(extreme_codes, m_fit, group_size=17)
            dec_ext, _ = independent_decode_le8cpu3(blob)
            np.testing.assert_array_equal(dec_ext, np.asarray(extreme_codes, dtype=np.int64))

        # Précision extrême = 1 et P = 1 (0 octet émis dans rank_bytes)
        m_prec1 = entropy.ShellModel((1, 1), limit=1, mode=0)
        zeros_and_esc = [[0] * 8] * 5 + [[2, 2, 0, 0, 0, 0, 0, 0]]
        b_p1 = entropy.encode_e8(zeros_and_esc, m_prec1, group_size=4)
        np.testing.assert_array_equal(independent_decode_le8cpu3(b_p1)[0], np.asarray(zeros_and_esc))

        # Précision extrême = 16 et LMATv3 complet
        freqs_p16 = tuple(entropy.normalized_frequencies([1] * 256 + [60000], precision=16))
        m_p16 = entropy.ShellModel(freqs_p16, limit=128, mode=1)
        mat_codes = e8.ROOTS[:16].reshape(4, 4, 8)
        mat_steps = np.array([np.float16(0.25), np.float16(0.5), np.float16(1.0), np.float16(2.0)], dtype=np.float64)
        lmat_blob = entropy.encode_matrix(mat_codes, mat_steps, m_p16, group_size=8)
        w_ind, c_ind, s_ind = independent_decode_lmatv3(lmat_blob)
        np.testing.assert_array_equal(c_ind, mat_codes)
        np.testing.assert_array_equal(s_ind, mat_steps)
        np.testing.assert_allclose(w_ind, (mat_codes.reshape(4, 32) / 2.0) * mat_steps[:, None], atol=0.0)

    def test_03_codec_bugs_and_rehashed_header_fuzzing(self):
        # BUG-CODEC-01 : asymétrie m=0 et n=0 entre encode_matrix et decode_matrix
        blob_m0 = entropy.encode_matrix(np.zeros((0, 2, 8), dtype=np.int64), np.zeros((0,), dtype=np.float16))
        with self.assertRaises(ValueError):
            entropy.decode_matrix(blob_m0)
        blob_n0 = entropy.encode_matrix(np.zeros((2, 0, 8), dtype=np.int64), np.ones((2,), dtype=np.float16))
        with self.assertRaises(ValueError):
            entropy.decode_matrix(blob_n0)

        # BUG-CODEC-02 : troncature silencieuse float64 -> FP16 (+98,68 % de distorsion sur pas subnormal !)
        codes_1 = e8.ROOTS[:2].reshape(1, 2, 8)
        subnormal_step = np.array([3.0e-8], dtype=np.float64)
        intended_w = (codes_1.reshape(1, 16) / 2.0) * subnormal_step[:, None]
        dec_w, _, _, _ = entropy.decode_matrix(entropy.encode_matrix(codes_1, subnormal_step))
        self.assertGreater(np.linalg.norm(dec_w - intended_w) / np.linalg.norm(intended_w), 0.98)

        # BUG-E8-01 : asymétrie ±1e8 de nearest, crash TypeError de valid sur bigint, et booléens acceptés
        _ = e8.nearest([1e8, 0, 0, 0, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            e8.nearest([-1e8, 0, 0, 0, 0, 0, 0, 0])
        with self.assertRaises(TypeError):
            e8.valid([1 << 64] * 8)
        self.assertTrue(e8.valid([True] * 8))
        self.assertEqual(e8.shell_size(True), 240)
        with self.assertRaises(ValueError):
            e8.shell_size(np.int64(1))

        # Fuzzing d'en-têtes et de sections avec SHA-256 recalculé valide
        valid_blob = entropy.encode_e8(e8.ROOTS[:16])
        body = bytearray(valid_blob[:-32])

        def rehash(b_arr: bytearray) -> bytes:
            return bytes(b_arr) + hashlib.sha256(bytes(b_arr)).digest()

        # Fréquence nulle avec somme = 2^prec
        b_zero_f = body.copy()
        f0, f1 = struct.unpack_from("<II", b_zero_f, 48)
        struct.pack_into("<II", b_zero_f, 48, 0, f0 + f1)
        with self.assertRaises(ValueError):
            entropy.decode_e8(rehash(b_zero_f))

        # Dépassement de max_vectors
        b_big_n = body.copy()
        struct.pack_into("<Q", b_big_n, 16, 5_000_000)
        with self.assertRaises(ValueError):
            entropy.decode_e8(rehash(b_big_n))

        # Mot de rang >= P
        _, mode, limit, _, _, _, _, la, lb, _ = IND_HDR.unpack_from(body, 0)
        nf = limit * (2 if mode else 1) + 1
        rk_start = 48 + 4 * nf + la
        b_bad_rk = body.copy()
        for pos in range(rk_start, rk_start + lb):
            b_bad_rk[pos] = 0xFF
        with self.assertRaises(ValueError):
            entropy.decode_e8(rehash(b_bad_rk))

    def test_04_length_invariance_breaks_and_quadratic_bigint_cost(self):
        # Rupture 1 : Mode 1, même coquille m=1, fréquences rANS identiques (8026 == 8026), mais p=0 -> p=1 (+3 o)
        c_eq = [0] * 129
        c_eq[2] = 5000
        c_eq[3] = 5000
        mod_m1 = entropy.ShellModel(tuple(entropy.normalized_frequencies(c_eq, 14)), limit=64, mode=1)
        b_p0 = entropy.encode_e8([e8.unrank(1, 0, 0)] * 128, mod_m1, group_size=128)
        b_p1 = entropy.encode_e8([e8.unrank(1, 0, 1)] * 128, mod_m1, group_size=128)
        self.assertEqual((len(b_p0), len(b_p1)), (725, 728))

        # Rupture 2 : Permutation adjacente d'UN ESC et d'UN non-ESC change la taille rANS `la` (135 vs 134 o)
        mod_swap = entropy.ShellModel((537, 3847, 8000, 4000), limit=3, mode=0)
        v0 = (0, 0, 0, 0, 0, 0, 0, 0)
        v2 = (4, 0, 0, 0, 0, 0, 0, 0)
        vesc = (200, 200, 0, 0, 0, 0, 0, 0)
        self.assertEqual(len(entropy.encode_e8([v0, v2, vesc], mod_swap, 128)), 135)
        self.assertEqual(len(entropy.encode_e8([v0, vesc, v2], mod_swap, 128)), 134)

        # Rupture 3 : Permutation de non-ESC entre groupes de rang (348 vs 347 o)
        mod_g2 = entropy.ShellModel(tuple(entropy.normalized_frequencies([5000, 0, 5000] + [0] * 62, 14)), limit=64, mode=0)
        self.assertEqual(len(entropy.encode_e8([v2, v0, v2, v0], mod_g2, group_size=2)), 348)
        self.assertEqual(len(entropy.encode_e8([v2, v2, v0, v0], mod_g2, group_size=2)), 347)

        # Vérification du mot géant de 118 264 bits autorisé par group_size=4096, m=127
        m127 = e8.shell_size(127)
        p4096 = m127 ** 4096
        self.assertEqual(p4096.bit_length(), 118264)

    def test_05_ill_conditioned_ldl_and_dense_kronecker_equivalence(self):
        for n, cond in ((16, 1e8), (24, 1e10), (24, 1e12)):
            h = _make_spd_with_cond(n, cond, 3)
            u, d = linalg.upper_ldl(h)
            u_inv = linalg.inverse_unit_upper(u)
            self.assertLess(np.max(np.abs(u @ u_inv - np.eye(n))), 1e-11)
            self.assertLess(np.linalg.norm((u * d[None, :]) @ u.T - h) / np.linalg.norm(h), 1e-12)

        for m, n, cond_g, cond_h, lam in ((5, 16, 25.0, 40.0, 0.03), (17, 24, 50.0, 80.0, 0.025)):
            ri = np.arange(1, m + 1, dtype=np.float64)[:, None]
            cj = np.arange(1, n + 1, dtype=np.float64)[None, :]
            w = 0.7 * np.cos(ri * (cj + 1.0) * 0.731) - 0.4 * np.sin((ri + 2.0) * cj * 1.17)
            g = _make_spd_with_cond(m, cond_g, 7)
            h = _make_spd_with_cond(n, cond_h, 11)
            model = entropy.ShellModel.gaussian(1.25)

            q_wave = quantization.quantize_ldlq(w, h, 0.8, lam, model, g)
            uk, fac_k = linalg.block_ldl(np.kron(g, h), 8)
            wf = w.ravel()
            qf = np.zeros_like(wf)
            codes_k = np.zeros((m, n // 8, 8), dtype=np.int64)
            for idx in range(0, m * n, 8):
                row = idx // n
                b = (idx % n) // 8
                inv_k, metric_k = fac_k[idx // 8]
                center_k = wf[idx:idx + 8] + ((wf[:idx] - qf[:idx]) @ uk[:idx, idx:idx + 8]) @ inv_k
                code_k, _ = quantization.select_code(center_k, q_wave.steps[row], metric_k, lam, model)
                codes_k[row, b] = code_k
                qf[idx:idx + 8] = code_k * q_wave.steps[row] / 2.0
            np.testing.assert_array_equal(q_wave.codes, codes_k)
            np.testing.assert_allclose(q_wave.weights.ravel(), qf, atol=1e-14)

            # Preuve du terme parasite intra-bloc si l'on applique (ui - I) à E_full post-quantification
            uo, _ = linalg.upper_ldl(g)
            ui, fac_h = linalg.block_ldl(h, 8)
            e_full = w - q_wave.weights
            spurious = ((uo - np.eye(m)).T @ e_full + (uo - np.eye(m)).T @ e_full @ (ui - np.eye(n)) + e_full @ (ui - np.eye(n)))[0, :8] @ fac_h[0][0]
            self.assertGreater(np.max(np.abs(spurious)), 1e-2)

    def test_06_checked_spd_asymmetry_four_bugs(self):
        # 1) BUG-LIN-01a : checked_spd accepte A mais upper_ldl lève LinAlgError
        a_asym = np.array([[1.0, 1.0 + 4.5e-12], [1.0 - 4.5e-12, 1.0 - 8.5e-12]], dtype=np.float64)
        linalg.checked_spd(a_asym)
        with self.assertRaises(np.linalg.LinAlgError):
            linalg.upper_ldl(a_asym)

        # 2) BUG-LIN-01b : crash ArithmeticError dans refine_quadratic
        n = 16
        h_crash = np.full((n, n), 1e6, dtype=np.float64) + 1e-4 * np.eye(n)
        for i in range(8):
            for j in range(8, 16):
                sgn = 1.0 if (i + j) % 2 == 0 else -1.0
                h_crash[i, j] += sgn * 0.49e-5
                h_crash[j, i] -= sgn * 0.49e-5
        linalg.checked_spd(h_crash)
        linalg.upper_ldl(h_crash)
        codes = np.zeros((1, 2, 8), dtype=np.int64)
        steps = np.array([np.float16(0.01)], dtype=np.float64)
        mod = entropy.ShellModel.fit(codes.reshape(-1, 8))
        init_c = quantization.Quantized(np.zeros((1, n)), codes, steps, mod, entropy.encode_matrix(codes, steps, mod))
        w_c = np.zeros((1, n), dtype=np.float64)
        w_c[0, 8:16] = np.array([50.0 if j % 2 == 0 else -50.0 for j in range(8, 16)])
        with self.assertRaises(ArithmeticError):
            quantization.refine_quadratic(w_c, init_c, h_crash, rate_lambda=0.0, sweeps=1, neighbors=True, preserve_shell=False)

        # 3) BUG-LIN-01c : hausse silencieuse non monotone de quadratic_objective
        h_silent = np.full((n, n), 100.0, dtype=np.float64) + np.eye(n)
        for i in range(8):
            for j in range(8, 16):
                sgn = 1.0 if (i + j) % 2 == 0 else -1.0
                h_silent[i, j] += sgn * 4.9e-10
                h_silent[j, i] -= sgn * 4.9e-10
        linalg.checked_spd(h_silent)
        steps_s = np.array([1.0], dtype=np.float64)
        init_s = quantization.Quantized(np.zeros((1, n)), codes, steps_s, mod, entropy.encode_matrix(codes, steps_s, mod))
        w_s = np.zeros((1, n), dtype=np.float64)
        w_s[0, 0] = 0.5 - 1e-7
        w_s[0, 1] = -0.5 + 1e-7
        w_s[0, 8:16] = np.array([-500.0 if j % 2 == 0 else 500.0 for j in range(8, 16)])
        _, log_s = quantization.refine_quadratic(w_s, init_s, h_silent, rate_lambda=0.0, sweeps=1, neighbors=True, preserve_shell=False)
        self.assertGreater(log_s["objective_trace_fixed_table"][1], log_s["objective_trace_fixed_table"][0])

        # 4) BUG-LIN-01d : transposée dans fresh_quadratic_target
        h_tr = np.array([[1e6 + 1.0, 1e6 + 4.9e-6], [1e6 - 4.9e-6, 1e6 + 1.0]], dtype=np.float64)
        b_tr = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
        tgt_tr, _ = calibration.fresh_quadratic_target(np.zeros((2, 2)), b_tr, h_tr, np.eye(2), trust_radius=1e9)
        self.assertLess(np.max(np.abs(tgt_tr @ h_tr.T + b_tr)), 1e-10)
        self.assertGreater(np.max(np.abs(tgt_tr @ h_tr + b_tr)), 9e-6)

    def test_07_bgcc_vectorized_equivalence_and_pcg_nan_bugs(self):
        rng = np.random.default_rng(77)
        m, n, out_d, N = 3, 8, 5, 20
        anchor = rng.normal(size=(m, n))
        y = rng.normal(size=(out_d, N))
        down = np.array([[1.0, 2.0, -1.0], [0.5, 1.0, -0.5], [2.0, 0.0, 1.0], [-1.0, 1.0, 0.5], [0.0, 0.0, 0.0]])

        res_w0, cg0 = calibration.bgcc_output_cg(anchor, rng.normal(size=(n, N)), np.zeros((m, N)), down, y, ridge=0.05)
        self.assertTrue(cg0.converged)
        self.assertEqual(cg0.iterations, 0)
        np.testing.assert_allclose(res_w0, anchor, atol=1e-15)

        xq_r1 = rng.normal(size=(n, 1)) @ rng.normal(size=(1, N))
        gates = linalg.silu(rng.normal(size=(m, N)))
        rho = 0.05
        sol_cg, cg_r1 = calibration.bgcc_output_cg(anchor, xq_r1, gates, down, y, ridge=rho, rtol=1e-11, maxiter=200)
        a_vec = rho * np.eye(m * n)
        dtd = down.T @ down
        for t in range(N):
            a_vec += np.kron(dtd * np.outer(gates[:, t], gates[:, t]), np.outer(xq_r1[:, t], xq_r1[:, t])) / N
        r0 = y - down @ (gates * (anchor @ xq_r1))
        b_vec = ((gates * (down.T @ r0)) @ xq_r1.T / N).ravel()
        sol_dense = anchor + np.linalg.solve(a_vec, b_vec).reshape(m, n)
        self.assertTrue(cg_r1.converged)
        np.testing.assert_allclose(sol_cg, sol_dense, atol=1e-9)

        # BUG-LIN-02 : sortie prématurée de pcg à k=20 < maxiter=200 avec converged=False
        d_ill = np.array([[1.0, -1.0, 0.5, -0.5]])
        g_ill = np.array([[1.0, 0.9, 1.1], [1.0, 0.9, 1.1], [0.8, 1.2, 0.7], [0.8, 1.2, 0.7]])
        x_ill = np.array([[1.0, -0.5, 0.3], [0.2, 1.1, -0.7]])
        mv, diag = calibration.bgcc_output_operator(x_ill, g_ill, d_ill, ridge=1e-11)
        rhs = np.array([[1.0, 1.0], [0.5, -0.3], [-0.7, -0.7], [0.2, 0.4]])
        premature = linalg.pcg(mv, rhs, diag, rtol=1e-9, maxiter=200)
        self.assertLess(premature.iterations, 30)
        self.assertFalse(premature.converged)
        self.assertGreater(premature.relative_residual, 1e-6)

        # BUG-LIN-03 : sous-flux float64 sur portes quasi-nulles levant ValueError("not numerically SPD")
        mv_tiny, diag_tiny = calibration.bgcc_output_operator(np.ones((2, 2)), np.full((2, 2), 1e-160), np.eye(2), ridge=1e8)
        with self.assertRaisesRegex(ValueError, "not numerically SPD"):
            linalg.pcg(mv_tiny, np.full((2, 2), 1e-160), diag_tiny, rtol=1e-9, maxiter=20)

        # BUG-CAL-01 : ridge=NaN et trust_radius=NaN non rejetés
        w_nan, _ = calibration.asymmetric_target(np.ones((2, 8)), np.ones((8, 4)), np.ones((8, 4)), float("nan"))
        self.assertTrue(np.all(np.isnan(w_nan)))
        _, info_tr = calibration.fresh_quadratic_target(np.zeros((2, 8)), np.ones((2, 8)), np.eye(8), np.eye(2), float("nan"))
        self.assertEqual(info_tr["scale"], 1.0)

    def test_08_rdoq_polish_metadata_bug_and_byte_inflation_counterexamples(self):
        # BUG-POL-01 : table_refitted_after_sweep vaut True même quand preserve_shell=True !
        w0 = np.ones((2, 8), dtype=np.float64)
        init0 = quantization.quantize_ldlq(w0, np.eye(8), 0.8, 0.0)
        _, info_iso = quantization.refine_quadratic(w0, init0, np.eye(8), preserve_shell=True)
        self.assertTrue(info_iso["shell_preserved"])
        self.assertTrue(info_iso["table_refitted_after_sweep"])

        # Contre-exemple 1 : refine_quadratic baisse l'objectif à table fixe mais AUGMENTE len(blob) (491 -> 492 o)
        m, n = 8, 32
        idx = np.arange(n)
        h = np.where(idx[:, None] == idx[None, :], 1.05, 0.75 * ((-0.85) ** np.abs(idx[:, None] - idx[None, :])))
        ri = np.arange(1, m + 1, dtype=np.float64)[:, None]
        cj = np.arange(1, n + 1, dtype=np.float64)[None, :]
        w = np.sin(ri * (cj + 2.0) * 1.71) + 0.6 * np.cos((ri + 1.0) * cj * 2.39)
        q_init = quantization.quantize_ldlq(w, h, step_multiplier=0.9, rate_lambda=0.0001)
        q_ref, log_q = quantization.refine_quadratic(w, q_init, h, rate_lambda=0.0001, sweeps=2, neighbors=True, preserve_shell=False)
        self.assertLess(log_q["objective_trace_fixed_table"][-1], log_q["objective_trace_fixed_table"][0])
        self.assertGreater(len(q_ref.artifact), len(q_init.artifact))

        # Contre-exemple 2 : refine_swiglu_up baisse l'objectif de > 53 % mais AUGMENTE len(blob) de +8 o (493 -> 501 o)
        d_out, N = 12, 64
        jn = np.arange(1, n + 1, dtype=np.float64)[:, None]
        tn = np.arange(1, N + 1, dtype=np.float64)[None, :]
        rm = np.arange(1, d_out + 1, dtype=np.float64)[:, None]
        xf = np.sin(jn * tn * 0.61) + 0.4 * np.cos((jn + 1.0) * (tn + 2.0) * 1.13)
        xq = xf + 0.35 * np.sin((jn + 4.0) * (tn + 1.0) * 2.07)
        wg = 0.45 * np.cos(ri * (cj + 1.0) * 1.31)
        wu = 0.45 * np.sin((ri + 1.0) * cj * 0.97)
        down = 0.65 * np.sin(rm * (ri.T + 2.0) * 1.53)
        hq = xq @ xq.T / N + 0.02 * np.eye(n)
        qg = quantization.quantize_ldlq(wg, hq, 0.85, 0.0001)
        qu0 = quantization.quantize_ldlq(wu, hq, 0.85, 0.0001)
        gates = linalg.silu(qg.weights @ xq)
        yc = down @ (linalg.silu(wg @ xf) * (wu @ xf))
        qu_ref, log_sw = calibration.refine_swiglu_up(qu0, xq, gates, down, yc, ridge=0.001, rate_lambda=0.0001, sweeps=2, preserve_shell=False)
        self.assertLess(log_sw["fixed_table_objective_trace"][-1], log_sw["fixed_table_objective_trace"][0])
        self.assertEqual((len(qu0.artifact), len(qu_ref.artifact)), (493, 501))

    def test_09_budget_dp_bruteforce_waterfill_and_mini_text_aux_audit(self):
        rng = np.random.default_rng(202604)
        for inst in range(20):
            groups = {
                f"G{g}": [budget.Option(f"g{g}_{k}", int(rng.integers(0, 10)), round(float(rng.normal()), 4)) for k in range(3)]
                for g in range(4)
            }
            fixed = int(rng.integers(0, 4))
            min_c = fixed + sum(min(o.bytes for o in v) for v in groups.values())
            max_c = fixed + sum(max(o.bytes for o in v) for v in groups.values())
            ceil_b = (min_c - 1) if inst >= 18 else int(rng.integers(min_c, max_c + 1))
            feas = [
                (sum(o.loss for o in p), fixed + sum(o.bytes for o in p))
                for p in itertools.product(*groups.values())
                if fixed + sum(o.bytes for o in p) <= ceil_b
            ]
            if not feas:
                with self.assertRaises(ValueError):
                    budget.allocate_options(groups, ceil_b, fixed)
            else:
                best_loss, best_cost = min(feas, key=lambda x: (x[0], x[1]))
                ans = budget.allocate_options(groups, ceil_b, fixed)
                self.assertAlmostEqual(ans["additive_loss"], best_loss, places=12)
                self.assertEqual(ans["bytes"], best_cost)

        with self.assertRaises(RuntimeError):
            budget.allocate_options(
                {f"g{i}": [budget.Option("0", 0, 0.0), budget.Option("1", 1 << i, -float(1 << i))] for i in range(6)},
                ceiling_bytes=100,
                max_states=10,
            )

        # BUG-BUD-01 : violation du budget dans continuous_waterfill + booléens acceptés + np.int64 rejeté + NaN
        r_viol = budget.continuous_waterfill([100.0, 200.0], [1.0, 4.0], 600.0 - 5e-6, floors=2.0, ceilings=8.0)
        self.assertGreater(float(np.dot([100.0, 200.0], r_viol)), 600.0 - 5e-6)
        self.assertEqual(budget.allocate_options({"g": [budget.Option("bool_true", True, 0.5)]}, ceiling_bytes=5)["bytes"], 1)
        with self.assertRaises(ValueError):
            budget.allocate_options({"g": [budget.Option("np_i64", np.int64(3), 0.5)]}, ceiling_bytes=5)
        self.assertTrue(math.isnan(budget.allocate_options({}, ceiling_bytes=float("nan"))["slack_bytes"]))

        # BUG-BUD-02 : Mini_text paie 84 066 560 o d'auxiliaires de Max (Vision+MTP inclus) et oublie protected_alignment
        f_v3 = budget_lumen_max_v3.facts()
        self.assertEqual(f_v3["Mini_text"]["scenarios"][0]["assumed_aux_bytes"], 84_066_560)
        self.assertEqual(f_v3["Max"]["protected_alignment_128B_extra_upper_bound"], 86_233)

    def test_10_metrics_duplicate_ids_negative_kl_underflow_and_fp16_inversion(self):
        # 1) BUG-MET-01 : IDs dupliqués [[0, 0]] -> KL strictement NÉGATIVE (-0,45496 nats) sur teacher == student !
        z_same = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        lp_same = metrics.log_softmax(z_same)
        dup_cache = {
            "ids": np.array([[0, 0]], dtype=np.int32),
            "log_probs": np.array([[lp_same[0, 0], lp_same[0, 0]]], dtype=np.float16),
            "log_tail_mass": np.array([np.logaddexp(lp_same[0, 1], lp_same[0, 2])], dtype=np.float32),
        }
        self.assertLess(metrics.grouped_kl_from_cache(dup_cache, z_same)[0], -0.45)
        self.assertLess(
            metrics.exact_grouped_kl(
                np.array([[0.0, -math.log(2.0), -math.log(2.0)]]),
                np.array([[math.log(4.0), -math.log(2.0), -math.log(2.0)]]),
                np.array([[0, 0]]),
            )[0],
            -0.01,
        )

        # 2) BUG-MET-02 : underflow catastrophique de exact_grouped_kl (> 54,6 nats d'erreur à -800 !)
        z_t = np.array([[0.0, 0.0, -50.0]])
        z_s = np.array([[0.0, -800.0, -800.0]])
        kl_cache = metrics.grouped_kl_from_cache(metrics.topk_cache(z_t, 2), z_s)[0]
        kl_oracle_broken = metrics.exact_grouped_kl(z_t, z_s, np.array([[0, 1]], dtype=np.int32))[0]
        self.assertAlmostEqual(kl_cache, 399.3068528, places=4)
        self.assertAlmostEqual(kl_oracle_broken, 344.6946168, places=4)

        # 3) BUG-MET-03 : overflow FP16 auto-infligé dans topk_cache faisant planter grouped_kl_from_cache
        z_deep = np.array([[0.0, -70000.0, -80000.0]])
        c_deep = metrics.topk_cache(z_deep, 2)
        with self.assertRaises(ValueError):
            metrics.grouped_kl_from_cache(c_deep, z_deep)

        # 4) Biais d'arrondi FP16 sur V = 248 320 (vocabulaire Qwen3.8-27B) : le vrai professeur a KL_cache > 0 !
        rng_v = np.random.default_rng(3827)
        z_qwen = rng_v.normal(scale=2.5, size=(1, 248_320))
        c_qwen = metrics.topk_cache(z_qwen, 128)
        kl_true_teacher_vs_cache = metrics.grouped_kl_from_cache(c_qwen, z_qwen)[0]
        self.assertGreater(kl_true_teacher_vs_cache, 1e-7)

    def test_11_accept_decoded_and_validate_capture_bypasses(self):
        # BUG-MET-04 : ceiling_bytes=NaN désactive le plafond et ignore les nouvelles sentinelles NaN !
        prev = metrics.ArtifactScore(b"abc", 1.0, {"s1": 0.5})
        cand_huge = metrics.ArtifactScore(b"x" * 10000, 0.9, {"s1": 0.5, "new_broken_domain": float("nan")})
        kept, info = metrics.accept_decoded(prev, cand_huge, ceiling_bytes=float("nan"))
        self.assertTrue(info["accepted"])
        self.assertIs(kept, cand_huge)

        # Candidat vide b"" accepté, et candidat 50 % plus petit à perte nulle (==) rejeté !
        kept_empty, info_empty = metrics.accept_decoded(prev, metrics.ArtifactScore(b"", 0.9, {"s1": 0.5}), ceiling_bytes=10)
        self.assertTrue(info_empty["accepted"])
        _, info_eq = metrics.accept_decoded(
            metrics.ArtifactScore(b"x" * 100, 1.0, {"s1": 0.5}),
            metrics.ArtifactScore(b"x" * 50, 1.0, {"s1": 0.5}),
            ceiling_bytes=100,
        )
        self.assertFalse(info_eq["accepted"])

        # BUG-CAP-01 : validate_capture accepte 0 token (5120, 0), splits vides [] et runtime_commit placeholder !
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            npz_p = td_p / "empty_tokens.npz"
            empty_arr = np.zeros((5120, 0), dtype=np.float32)
            np.savez(npz_p, X_fp_cal=empty_arr, X_q_cal=empty_arr, X_fp_select=empty_arr, X_q_select=empty_arr, X_fp_confirm=empty_arr, X_q_confirm=empty_arr)
            man_p = td_p / "man.json"
            man_p.write_text(json.dumps({
                "checkpoint": "Qwen/Qwen3.8-27B",
                "revision": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
                "runtime_commit": "A_REMPLIR_PAR_LE_COLLECTEUR",
                "activation_path": "x",
                "full_input_width": 5120.9,
                "splits": {"cal": [], "select": [], "confirm": []},
                "array_sha256": hashlib.sha256(npz_p.read_bytes()).hexdigest(),
                "source_kind": "captured_from_model_forward",
                "architecture_id": "A0",
                "weight_status": "x",
            }), encoding="utf-8")
            self.assertTrue(validate_capture.validate(npz_p, man_p)["schema_valid"])

    def test_12_protocol_byte_mismatch_and_original_directory_untouched(self):
        exp = json.loads((TARGET_V3_ROOT / "results" / "experiences_synthetiques.json").read_text(encoding="utf-8"))
        t601 = [t for t in exp["trials"] if t["seed"] == 601][0]
        self.assertEqual(t601["Qplus_search_control"]["bytes"], 1532)
        self.assertEqual(t601["families"]["asymmetric"]["bytes"], 1611)  # +79 octets (+5,16 %) pour asymmetric !

        iso = json.loads((TARGET_V3_ROOT / "results" / "experience_isocoquille.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(1 for t in iso["trials"] if t["accepted_on_select"]), 0)  # 0/5 acceptés sur sélection

        # Preuve finale que LUMEN_Q1_V3_CORRIGE est resté 100 % bit-à-bit identique à l'état initial
        self.assertEqual(_snapshot_tree(TARGET_V3_ROOT), INITIAL_SNAPSHOT)


if __name__ == "__main__":
    t0 = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentAuditV3Suite)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"\n[IndependentAuditV3] Ran {res.testsRun} test groups in {time.perf_counter() - t0:.3f}s — ok={res.wasSuccessful()}")
    if not res.wasSuccessful():
        raise SystemExit(1)
