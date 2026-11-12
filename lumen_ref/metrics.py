"""Stable log-space metrics and explicit, fail-closed artifact acceptance.

Top-K caches are approximations when rounded. Their actual array bytes and
precision are reported; they are not a substitute for the full KL. Masks may
use -inf for exact zero probability. +inf/NaN and empty distributions fail.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import numbers
import numpy as np
from .linalg import as_finite


def _logs(a, ndim=2):
    if np.iscomplexobj(a):
        raise ValueError("Real log-values required")
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != ndim or np.any(np.isnan(a)) or np.any(np.isposinf(a)):
        raise ValueError("Wrong shape, NaN or positive infinity")
    return a


def _logsumexp_rows(a):
    """All -inf rows return -inf without warnings; not normalizable logits."""
    a = _logs(a)
    if a.shape[1] == 0:
        return np.full(a.shape[0], -np.inf)
    mx = a.max(axis=1)
    good = np.isfinite(mx)
    out = np.full(a.shape[0], -np.inf)
    if np.any(good):
        shifted = a[good] - mx[good, None]
        out[good] = mx[good] + np.log(np.exp(shifted).sum(axis=1))
    return out


def log_softmax(z):
    z = _logs(z)
    if z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError("Empty conditional distribution")
    mx = z.max(axis=1, keepdims=True)
    if not np.all(np.isfinite(mx)):
        raise ValueError("A row has no unmasked finite logit")
    with np.errstate(over='ignore', invalid='ignore'):
        v = z - mx
    if np.any(np.isnan(v)) or np.any(np.isneginf(v) & np.isfinite(z)):
        raise ValueError("Logit range exceeds float64; rescale upstream")
    return v - np.log(np.exp(v).sum(axis=1, keepdims=True))


def _kl_normalized_logs(t, s):
    if t.shape != s.shape:
        raise ValueError("Different conditional distributions")
    p = np.exp(t)
    # Exact zero mass contributes zero even against zero student support.
    terms = np.zeros_like(t)
    active = p > 0
    with np.errstate(invalid='ignore', over='ignore'):
        terms[active] = p[active] * (t[active] - s[active])
    values = np.sum(terms, axis=1)
    if np.any(np.isnan(values)):
        raise ArithmeticError("Undefined KL terms")
    # Permit only roundoff-scale negativity; never conceal an invalid partition.
    finite_terms = np.where(np.isfinite(terms), np.abs(terms), 0.)
    tolerance = 64*np.finfo(np.float64).eps*np.maximum(1., finite_terms.sum(axis=1))
    if np.any(values < -tolerance):
        raise ArithmeticError("Significantly negative KL; invalid distribution or partition")
    return np.where((values < 0) & (values >= -tolerance), 0., values)


def kl_logits(teacher, student):
    if np.shape(teacher) != np.shape(student):
        raise ValueError("Different conditional distributions")
    return _kl_normalized_logs(log_softmax(teacher), log_softmax(student))


def _checked_topk_ids(ids, rows: int, vocab: int):
    a = np.asarray(ids)
    if a.dtype == np.bool_ or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("Top-K ids must be integers, not bool or floats")
    if (a.ndim != 2 or a.shape[0] != rows or not 1 <= a.shape[1] < vocab
            or np.any(a < 0) or np.any(a >= vocab)):
        raise ValueError("Invalid Top-K ids")
    if a.shape[1] > 1 and np.any(np.diff(np.sort(a, axis=1), axis=1) == 0):
        raise ValueError("Duplicate token ids in Top-K")
    return a


def _group_logs(lp, ids):
    top = np.take_along_axis(lp, ids, axis=1)
    mask = np.ones_like(lp, dtype=bool)
    np.put_along_axis(mask, ids, False, axis=1)
    tail = _logsumexp_rows(np.where(mask, lp, -np.inf))
    return np.concatenate([top, tail[:, None]], axis=1)


def _store_logs(a, dtype, overflow_policy):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float16'), np.dtype('float32'), np.dtype('float64')):
        raise ValueError("Unsupported cache dtype")
    finite = np.isfinite(a)
    if np.any(a[finite] < -np.finfo(dtype).max):
        if overflow_policy == 'raise':
            raise ValueError("Cache log-probability cannot be stored in requested dtype")
        for name in ('float32', 'float64'):
            candidate = np.dtype(name)
            if candidate.itemsize > dtype.itemsize and not np.any(a[finite] < -np.finfo(candidate).max):
                dtype = candidate
                break
    result = a.astype(dtype)
    if np.any(finite & ~np.isfinite(result)):
        raise ArithmeticError("Finite cache value became infinite")
    return result


def topk_cache(teacher, k, value_dtype=np.float16, overflow_policy='promote'):
    """Normalized log probabilities, int32 IDs, log tail mass (>= FP32).

    Ordinary values use requested precision. Extreme finite values promote the
    entire value array, or raise if requested; NO silent clipping. Exact masked
    probabilities remain -inf. Metadata counts ndarray bytes, not a file.
    """
    lp = log_softmax(teacher)
    vocab = lp.shape[1]
    if (isinstance(k, (bool, np.bool_)) or not isinstance(k, numbers.Integral)
            or not 1 <= k < vocab or vocab > np.iinfo(np.int32).max):
        raise ValueError("Require integer 1 <= k < vocabulary size <= int32 max")
    if overflow_policy not in ('promote', 'raise'):
        raise ValueError("Unknown overflow policy")
    ids = np.argsort(lp, axis=1, kind='stable')[:, -int(k):].astype(np.int32)
    grouped = _group_logs(lp, ids)
    vals = _store_logs(grouped[:, :-1], value_dtype, overflow_policy)
    # float64 cache is an oracle option: do not leave its tail rounded to FP32.
    tail_dtype = np.float64 if np.dtype(value_dtype) == np.dtype('float64') else np.float32
    tail = _store_logs(grouped[:, -1], tail_dtype, overflow_policy)
    return {'ids': ids, 'log_probs': vals, 'log_tail_mass': tail,
            'vocabulary_size': vocab, 'log_probs_dtype': str(vals.dtype),
            'tail_dtype': str(tail.dtype), 'cache_kind': 'normalized_log_probs_plus_tail',
            'actual_array_bytes': int(ids.nbytes + vals.nbytes + tail.nbytes),
            'requested_log_probs_dtype': str(np.dtype(value_dtype)),
            'rounding_bias_requires_validation': np.dtype(value_dtype) != np.dtype('float64')}


def grouped_kl_from_cache(cache, student):
    lp = log_softmax(student)
    ids = _checked_topk_ids(cache['ids'], len(lp), lp.shape[1])
    t = _logs(cache['log_probs'])
    tail = _logs(cache['log_tail_mass'], 1)
    if ids.shape != t.shape or tail.shape != (len(lp),):
        raise ValueError("Invalid cache arrays")
    if 'vocabulary_size' in cache and cache['vocabulary_size'] != lp.shape[1]:
        raise ValueError("Cache vocabulary mismatch")
    if np.any(t > 0) or np.any(tail > 0):
        raise ValueError("Cache must contain normalized log probabilities")
    raw = np.concatenate([t, tail[:, None]], axis=1)
    mass = _logsumexp_rows(raw)
    if not np.all(np.isfinite(mass)) or np.any(np.abs(mass) > 0.1):
        raise ValueError("Cache mass is malformed, beyond permitted storage rounding")
    # Explicit renormalization of the rounded teacher, NOT the exact teacher.
    return _kl_normalized_logs(log_softmax(raw), _group_logs(lp, ids))


def exact_grouped_kl(teacher, student, ids):
    a, b = log_softmax(teacher), log_softmax(student)
    if a.shape != b.shape:
        raise ValueError("Different conditional distributions")
    ids = _checked_topk_ids(ids, len(a), a.shape[1])
    return _kl_normalized_logs(_group_logs(a, ids), _group_logs(b, ids))


def fisher_probe(p, zeta):
    """1-D probability vector; zeta may batch probes. Moments are caller's job."""
    p = as_finite(p, 1); z = as_finite(zeta)
    if p.size == 0 or z.ndim < 1 or z.shape[-1] != len(p) or np.any(p < 0) or not np.isclose(p.sum(), 1.):
        raise ValueError("Invalid distribution/probe")
    s = np.sqrt(p)
    return s*z - np.sum(s*z, axis=-1, keepdims=True)*p


def assert_disjoint_splits(cal_ids, select_ids, confirm_ids):
    sets = [set(cal_ids), set(select_ids), set(confirm_ids)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Source/document/episode overlap between splits")


@dataclass(frozen=True)
class ArtifactScore:
    data: bytes
    primary: float
    sentinels: dict[str, float]  # all lower-is-better by contract

    def __post_init__(self):
        # Defensive copies prevent post-decision mutation of the accepted state.
        if not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise ValueError("Artifact must contain bytes")
        object.__setattr__(self, 'data', bytes(self.data))
        object.__setattr__(self, 'sentinels', MappingProxyType(dict(self.sentinels)))

    @property
    def sha256(self):
        return hashlib.sha256(self.data).hexdigest()


def accept_decoded(previous: ArtifactScore, candidate: ArtifactScore, ceiling_bytes: int,
                   required_sentinels=None, tolerances=None, *, validator=None):
    """Select a scored artifact; a supplied validator additionally decodes bytes.

    Scores must come from those EXACT bytes. Without a validator this is a
    comparator, not proof of decodability. Use compare_matrix_artifacts below
    when scores should be computed internally after real decoding.
    """
    if (isinstance(ceiling_bytes, (bool, np.bool_)) or
            not isinstance(ceiling_bytes, numbers.Integral) or ceiling_bytes <= 0):
        raise ValueError("ceiling_bytes must be a positive integer")
    ceiling_bytes = int(ceiling_bytes)
    required = (set(previous.sentinels) | set(candidate.sentinels)) if required_sentinels is None else set(required_sentinels)
    tol = {} if tolerances is None else dict(tolerances)
    if any(not isinstance(k, str) or not np.isfinite(v) or v < 0 for k, v in tol.items()):
        raise ValueError("Invalid sentinel tolerance")
    if not previous.data or len(previous.data) > ceiling_bytes or not np.isfinite(previous.primary):
        raise ValueError("Previous artifact is empty, infeasible or unscored")
    if any(not isinstance(k, str) or not np.isfinite(v) for k, v in previous.sentinels.items()):
        raise ValueError("Invalid previous sentinel")
    reason = []
    if not candidate.data:
        reason.append('empty_candidate')
    if len(candidate.data) > ceiling_bytes:
        reason.append('size_exceeded')
    if (not np.isfinite(candidate.primary) or candidate.primary > previous.primary or
            (candidate.primary == previous.primary and len(candidate.data) >= len(previous.data))):
        reason.append('no_primary_improvement')
    for k, v in candidate.sentinels.items():
        if not isinstance(k, str) or not np.isfinite(v):
            reason.append('invalid_sentinel:'+str(k))
    for key in required:
        if key not in candidate.sentinels or key not in previous.sentinels:
            reason.append('missing_sentinel:'+str(key)); continue
        c, p = candidate.sentinels[key], previous.sentinels[key]
        if not np.isfinite(c) or c > p + tol.get(key, 0.):
            reason.append('sentinel_regression:'+key)
    if validator is not None:
        if validator(previous.data) is False:
            raise ValueError("Previous artifact failed independent decoding")
        try:
            if validator(candidate.data) is False:
                reason.append('decode_failed')
        except (ValueError, ArithmeticError, OverflowError):
            reason.append('decode_failed')
    return (previous if reason else candidate), {
        'accepted': not reason, 'reasons': reason,
        'decoder_checked': validator is not None,
        'guarantee': 'measured_scores_and_bytes_only_not_generalization'}


def compare_matrix_artifacts(previous: bytes, candidate: bytes, evaluator, ceiling_bytes: int,
                             required_sentinels=None, tolerances=None, max_weights=8_000_000):
    """Decode each LMAT blob and score its weights. Returns selected bytes/trace.

    evaluator(weights) -> (primary_lower_is_better, sentinel_mapping).
    Invalid previous bytes are an error; invalid candidates retain previous.
    """
    from .entropy import decode_matrix
    wp, _, _, _ = decode_matrix(previous, max_weights=max_weights)
    pp, sp = evaluator(wp)
    previous_score = ArtifactScore(previous, float(pp), sp)
    try:
        wc, _, _, _ = decode_matrix(candidate, max_weights=max_weights)
        if wc.shape != wp.shape:
            raise ValueError('Candidate matrix changed the compared interface')
        pc, sc = evaluator(wc)
        candidate_score = ArtifactScore(candidate, float(pc), sc)
    except (ValueError, ArithmeticError, OverflowError) as exc:
        # Check the budget/config even when the candidate does not decode.
        accept_decoded(previous_score, ArtifactScore(b'', np.inf, {}), ceiling_bytes,
                       required_sentinels, tolerances)
        return previous, {'accepted': False, 'reasons': ['decode_or_evaluation_failed'], 'detail': str(exc), 'decoder_checked': True}
    chosen, trace = accept_decoded(previous_score, candidate_score, ceiling_bytes,
                                  required_sentinels, tolerances)
    trace['decoder_checked'] = True
    trace['previous_primary'] = float(pp)
    trace['candidate_primary'] = float(pc)
    trace['selected_sha256'] = chosen.sha256
    return chosen.data, trace
