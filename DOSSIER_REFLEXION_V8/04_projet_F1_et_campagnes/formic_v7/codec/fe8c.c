/* fe8c.c -- range coder core for the FORMIC E8 codec (fcodec.py). Single-threaded, ctypes API.
 * Coordinates are "doubled E8" integers u: all same parity c, sum(u) % 4 == 0.
 * Mode 0 (SHELL): per vector  shell symbol m (categorical, ESC = raw 8 x 32 bit),
 *                 then the point inside shell m coded EXACTLY uniformly (enumerative, via count table T).
 * Mode 1 (PROD):  per vector  coset c (categorical), v_i=(u_i-c)>>1 for i<7 with table[c] (ESC = raw 32 bit),
 *                 last coordinate: parity p of v_7 is implied by sum(v) even; code k=v_7>>1 with table[2+2c+p].
 * Range coder: 64-bit low/range, byte output, carry propagated in the output buffer.
 */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define TOP (1ULL << 56)
#define LUTB 12

typedef struct { uint8_t *out; int64_t pos, cap; uint64_t low, range; int err; } Enc;
typedef struct { const uint8_t *in; int64_t pos, len; uint64_t x, range; } Dec;

static inline void enc_put(Enc *e, uint64_t cum, uint64_t f, uint64_t r) {
    uint64_t add = r * cum;
    e->low += add;
    if (e->low < add) {               /* carry */
        int64_t i = e->pos - 1;
        while (i >= 0 && e->out[i] == 0xFF) { e->out[i] = 0; i--; }
        if (i >= 0) e->out[i]++;
    }
    e->range = r * f;
    while (e->range < TOP) {
        if (e->pos >= e->cap) { e->err = 1; e->pos = 0; }
        e->out[e->pos++] = (uint8_t)(e->low >> 56);
        e->low <<= 8; e->range <<= 8;
    }
}
static inline void enc_sym(Enc *e, uint64_t cum, uint64_t f, uint64_t tot) { enc_put(e, cum, f, e->range / tot); }
static inline void enc_symP(Enc *e, uint64_t cum, uint64_t f, int P) { enc_put(e, cum, f, e->range >> P); }
static inline void enc_raw32(Enc *e, uint32_t v) { enc_symP(e, v >> 16, 1, 16); enc_symP(e, v & 0xFFFF, 1, 16); }
static void enc_flush(Enc *e) {
    for (int i = 0; i < 8; i++) {
        if (e->pos >= e->cap) { e->err = 1; return; }
        e->out[e->pos++] = (uint8_t)(e->low >> 56); e->low <<= 8;
    }
}
static inline uint8_t dnext(Dec *d) { return d->pos < d->len ? d->in[d->pos++] : (d->pos++, 0); }
static void dec_init(Dec *d) { d->x = 0; for (int i = 0; i < 8; i++) d->x = (d->x << 8) | dnext(d); d->range = ~0ULL; }
static inline void dec_upd(Dec *d, uint64_t cum, uint64_t f, uint64_t r) {
    d->x -= r * cum; d->range = r * f;
    while (d->range < TOP) { d->x = (d->x << 8) | dnext(d); d->range <<= 8; }
}
static inline uint32_t dec_raw32(Dec *d) {
    uint64_t r = d->range >> 16, v = d->x / r; if (v > 0xFFFF) v = 0xFFFF; dec_upd(d, v, 1, r);
    uint64_t r2 = d->range >> 16, w = d->x / r2; if (w > 0xFFFF) w = 0xFFFF; dec_upd(d, w, 1, r2);
    return (uint32_t)((v << 16) | w);
}

/* categorical tables: cum[off .. off+nsym], total 2^P; lut built for decoding */
typedef struct { const uint32_t *cum; int32_t lo, nsym; uint16_t *lut; } Tab;

static void build_tabs(Tab *t, int nt, const int32_t *desc, const uint32_t *cums, int P, uint16_t *lutmem) {
    for (int i = 0; i < nt; i++) {
        t[i].lo = desc[3 * i]; t[i].nsym = desc[3 * i + 1]; t[i].cum = cums + desc[3 * i + 2];
        t[i].lut = lutmem ? lutmem + ((int64_t)i << LUTB) : NULL;
        if (lutmem) {
            int s = 0, sh = P - LUTB;
            for (int j = 0; j < (1 << LUTB); j++) {
                uint32_t v = (uint32_t)j << sh;
                while (s < t[i].nsym - 1 && t[i].cum[s + 1] <= v) s++;
                t[i].lut[j] = (uint16_t)s;
            }
        }
    }
}
static inline int dec_tab(Dec *d, const Tab *t, int P) {
    uint64_t r = d->range >> P, v = d->x / r;
    if (v >= (1ULL << P)) v = (1ULL << P) - 1;
    int s = t->lut[v >> (P - LUTB)];
    while (t->cum[s + 1] <= v) s++;
    dec_upd(d, t->cum[s], t->cum[s + 1] - t->cum[s], r);
    return s;
}
static inline void enc_tab(Enc *e, const Tab *t, int s, int P) { enc_symP(e, t->cum[s], t->cum[s + 1] - t->cum[s], P); }

/* ---------------- count table T[c][k][r][s] ---------------- */
static uint64_t *T = NULL; static int TR = 0;   /* r in [0, TR] */
#define TIDX(c, k, r, s) ((((int64_t)(c) * 9 + (k)) * (TR + 1) + (r)) * 4 + (s))

int fe8_init_counts(int rmax) {
    if (T && TR >= rmax) return 0;
    free(T); TR = rmax;
    T = (uint64_t *)calloc((size_t)2 * 9 * (TR + 1) * 4, sizeof(uint64_t));
    if (!T) return -1;
    for (int c = 0; c < 2; c++) {
        T[TIDX(c, 0, 0, 0)] = 1;
        for (int k = 1; k <= 8; k++)
            for (int r = 0; r <= TR; r++)
                for (int s = 0; s < 4; s++) {
                    uint64_t acc = 0;
                    for (int a = c; a * a <= r; a += 2) {
                        acc += T[TIDX(c, k - 1, r - a * a, (s - a) & 3)];
                        if (a) acc += T[TIDX(c, k - 1, r - a * a, (s + a) & 3)];
                    }
                    T[TIDX(c, k, r, s)] = acc;
                }
    }
    return 0;
}
uint64_t fe8_count(int c, int k, int r, int s) { return (r <= TR) ? T[TIDX(c, k, r, s)] : 0; }

/* ---------------- stats ---------------- */
/* returns -1 if all vectors valid else index of first invalid vector.
 * shist[kmax+2]: m in [0,kmax] -> shist[m], else shist[kmax+1].
 * phist: 6 tables x (2W+2): [0]: coset (2 used), [1..2]: v|c, [3..6]: k|(c,p); index W+v, escape at 2W+1. */
int64_t fe8_stats(const int32_t *U, int64_t nvec, int kmax, int64_t *shist, int W, int64_t *phist) {
    int64_t L = 2 * W + 2;
    for (int64_t i = 0; i < nvec; i++) {
        const int32_t *u = U + 8 * i;
        int c = u[0] & 1; int64_t sum = 0, sq = 0; int bad = 0;
        int big = 0;
        for (int j = 0; j < 8; j++) { bad |= ((u[j] & 1) != c); sum += u[j]; sq += (int64_t)u[j] * u[j]; big |= (u[j] > 65535 || u[j] < -65535); }
        if (bad || (sum & 3)) return i;
        int64_t m = big ? (int64_t)kmax + 1 : (sq >> 3);   /* avoid int64 overflow for |u| ~ 2^31 */
        shist[m <= kmax ? m : kmax + 1]++;
        phist[c]++;
        int64_t vs = 0;
        for (int j = 0; j < 7; j++) {
            int64_t v = ((int64_t)u[j] - c) >> 1; vs += v;
            phist[(1 + c) * L + ((v >= -W && v <= W) ? W + v : 2 * W + 1)]++;
        }
        int64_t v7 = ((int64_t)u[7] - c) >> 1; int p = (int)(vs & 1);
        int64_t k = v7 >> 1;
        phist[(3 + 2 * c + p) * L + ((k >= -W && k <= W) ? W + k : 2 * W + 1)]++;
    }
    return -1;
}

/* ---------------- shell-mode in-shell coding ---------------- */
static inline void enc_inshell(Enc *e, const int32_t *u, int m) {
    int r = 8 * m;
    uint64_t n0 = T[TIDX(0, 8, r, 0)], n1 = T[TIDX(1, 8, r, 0)];
    int c = u[0] & 1;
    if (n0 && n1) enc_sym(e, c ? n0 : 0, c ? n1 : n0, n0 + n1);
    int s = 0;
    for (int j = 0; j < 8; j++) {
        int k = 8 - j, a = u[j];
        uint64_t tot = T[TIDX(c, k, r, s)], cum = 0, f = 0;
        /* candidate order: c, -c(if nonzero), c+2, -(c+2), ... */
        for (int b = c; b * b <= r; b += 2) {
            uint64_t fp = T[TIDX(c, k - 1, r - b * b, (s - b) & 3)];
            if (a == b) { f = fp; break; }
            cum += fp;
            if (b) {
                uint64_t fm = T[TIDX(c, k - 1, r - b * b, (s + b) & 3)];
                if (a == -b) { f = fm; break; }
                cum += fm;
            }
        }
        if (f != tot) enc_sym(e, cum, f, tot);
        r -= a * a; s = (s - a) & 3;
    }
}
static inline void dec_inshell(Dec *d, int32_t *u, int m) {
    int r = 8 * m;
    uint64_t n0 = T[TIDX(0, 8, r, 0)], n1 = T[TIDX(1, 8, r, 0)];
    int c;
    if (n0 && n1) {
        uint64_t rr = d->range / (n0 + n1), v = d->x / rr;
        c = v >= n0; dec_upd(d, c ? n0 : 0, c ? n1 : n0, rr);
    } else c = n1 ? 1 : 0;
    int s = 0;
    for (int j = 0; j < 8; j++) {
        int k = 8 - j;
        uint64_t tot = T[TIDX(c, k, r, s)];
        uint64_t rr = 0, v = 0;
        int need = 1;
        int a = 0; uint64_t cum = 0, f = 0;
        for (int b = c; b * b <= r; b += 2) {
            uint64_t fp = T[TIDX(c, k - 1, r - b * b, (s - b) & 3)];
            if (fp == tot) { a = b; f = fp; need = 0; break; }
            if (!rr) { rr = d->range / tot; v = d->x / rr; if (v >= tot) v = tot - 1; }
            if (v < cum + fp) { a = b; f = fp; break; }
            cum += fp;
            if (b) {
                uint64_t fm = T[TIDX(c, k - 1, r - b * b, (s + b) & 3)];
                if (fm == tot) { a = -b; f = fm; need = 0; break; }
                if (v < cum + fm) { a = -b; f = fm; break; }
                cum += fm;
            }
        }
        if (need) dec_upd(d, cum, f, rr);
        u[j] = a; r -= a * a; s = (s - a) & 3;
    }
}

/* ---------------- encode / decode ---------------- */
/* desc: ntab x (lo, nsym, cum_offset); ESC is the last symbol of each table. */
int64_t fe8_encode(int mode, const int32_t *U, int64_t nvec, int ntab, const int32_t *desc, const uint32_t *cums,
                   int P, uint8_t *out, int64_t cap) {
    Tab t[8]; build_tabs(t, ntab, desc, cums, P, NULL);
    Enc e = { out, 0, cap, 0, ~0ULL, 0 };
    for (int64_t i = 0; i < nvec && !e.err; i++) {
        const int32_t *u = U + 8 * i;
        if (mode == 0) {
            int64_t sq = 0; int big = 0;
            for (int j = 0; j < 8; j++) { sq += (int64_t)u[j] * u[j]; big |= (u[j] > 65535 || u[j] < -65535); }
            int esc = t[0].nsym - 1; int64_t m = big ? esc : (sq >> 3);
            if (m < esc) { enc_tab(&e, &t[0], (int)m, P); enc_inshell(&e, u, (int)m); }
            else { enc_tab(&e, &t[0], esc, P); for (int j = 0; j < 8; j++) enc_raw32(&e, (uint32_t)u[j]); }
        } else {
            int c = u[0] & 1; int64_t vs = 0;
            enc_tab(&e, &t[0], c, P);
            const Tab *tc = &t[1 + c];
            for (int j = 0; j < 7; j++) {
                int64_t v = ((int64_t)u[j] - c) >> 1, s = v - tc->lo; vs += v;
                if (s >= 0 && s < tc->nsym - 1) enc_tab(&e, tc, (int)s, P);
                else { enc_tab(&e, tc, tc->nsym - 1, P); enc_raw32(&e, (uint32_t)(int32_t)v); }
            }
            int p = (int)(vs & 1); int64_t k = (((int64_t)u[7] - c) >> 1) >> 1;
            const Tab *tk = &t[3 + 2 * c + p]; int64_t s = k - tk->lo;
            if (s >= 0 && s < tk->nsym - 1) enc_tab(&e, tk, (int)s, P);
            else { enc_tab(&e, tk, tk->nsym - 1, P); enc_raw32(&e, (uint32_t)(int32_t)k); }
        }
    }
    if (!e.err) enc_flush(&e);
    return e.err ? -1 : e.pos;
}

int fe8_decode(int mode, const uint8_t *in, int64_t len, int64_t nvec, int ntab, const int32_t *desc,
               const uint32_t *cums, int P, int32_t *U) {
    uint16_t *lut = (uint16_t *)malloc((size_t)ntab << LUTB << 1);
    if (!lut) return -2;
    Tab t[8]; build_tabs(t, ntab, desc, cums, P, lut);
    Dec d = { in, 0, len, 0, 0 }; dec_init(&d);
    for (int64_t i = 0; i < nvec; i++) {
        int32_t *u = U + 8 * i;
        if (mode == 0) {
            int s = dec_tab(&d, &t[0], P);
            if (s < t[0].nsym - 1) dec_inshell(&d, u, s);
            else for (int j = 0; j < 8; j++) u[j] = (int32_t)dec_raw32(&d);
        } else {
            int c = dec_tab(&d, &t[0], P); int64_t vs = 0;
            const Tab *tc = &t[1 + c];
            for (int j = 0; j < 7; j++) {
                int s = dec_tab(&d, tc, P); int64_t v;
                if (s < tc->nsym - 1) v = tc->lo + s; else v = (int32_t)dec_raw32(&d);
                vs += v; u[j] = (int32_t)(2 * v + c);
            }
            int p = (int)(vs & 1); const Tab *tk = &t[3 + 2 * c + p];
            int s = dec_tab(&d, tk, P); int64_t k;
            if (s < tk->nsym - 1) k = tk->lo + s; else k = (int32_t)dec_raw32(&d);
            u[7] = (int32_t)(2 * (2 * k + p) + c);
        }
    }
    free(lut);
    return d.pos > d.len + 8 ? -1 : 0;   /* reading far past the end = corrupt stream */
}
