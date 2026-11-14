"""Real-weight check: ideal-bit accounting vs physical codec size, CPU only."""
import os, sys, json, io, contextlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
from safetensors import safe_open
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lab'))
sys.path.insert(0, '/workspace/LUMEN_Q1_V3.1/FORMIC_RESEARCH_INFRA_V1/vendor')
import formic_q as fq
from lumen_ref.entropy import ShellModel, encode_matrix, decode_matrix

D = '/root/models/qwen3.8-27b/'
wm = json.load(open(D + 'model.safetensors.index.json'))['weight_map']
MULT = 0.5917; FIXED = 656
L = 'model.language_model.layers.'
cases = [(L+'30.mlp.gate_proj.weight', 0, 64), (L+'30.mlp.down_proj.weight', 0, 16),
         (L+'30.linear_attn.in_proj_qkv.weight', 0, 64), (L+'31.self_attn.q_proj.weight', 0, 64),
         (L+'31.self_attn.o_proj.weight', 0, 48), ('model.language_model.embed_tokens.weight', 1000, 1064),
         ('lm_head.weight', 1000, 1064)]
out = []; buf = io.StringIO()
def P(s):
    print(s, flush=True); buf.write(s + "\n")
for name, a, b in cases:
    if b - a > 64 and a == 0: pass
    with safe_open(D + wm[name], "pt") as f:
        sl = f.get_slice(name); shape = sl.get_shape(); W = sl[a:b].float()
    m, n = W.shape
    r = fq.row_steps(W, MULT)
    Q, hist = fq.quantize_rtn(W, r)
    rn = r.double().numpy()
    Qn = Q.numpy()
    codes = np.rint(2 * Qn.astype(np.float64) / rn[:, None]).astype(np.int64).reshape(m, n // 8, 8)
    recon = (torch.from_numpy(codes).double() * 0.5 * torch.from_numpy(rn)[:, None, None]).float().reshape(m, n).numpy()
    valid = bool(np.array_equal(recon, Qn))
    ideal = fq.bits_from_hist(hist)
    ideal2 = fq.bits_from_hist(fq.shell_hist(torch.from_numpy(codes)))
    model = ShellModel.fit(codes.reshape(-1, 8), limit=128, mode=0, precision=14)
    blob = encode_matrix(codes, rn, model, group_size=128)
    phys = 8 * len(blob)
    dec, dcodes, dsteps, _ = decode_matrix(blob, max_weights=m * n)
    dec = np.asarray(dec).reshape(m, n)
    maxdiff = float(np.max(np.abs(dec - Qn.astype(np.float64))))
    rt_codes = bool(np.array_equal(dcodes.reshape(codes.shape), codes)); rt_steps = bool(np.array_equal(dsteps, rn))
    N = m * n
    prop = (phys - ideal - 16 * m - 8 * FIXED) / N
    sh = (codes.astype(np.float64) ** 2).sum(-1) / 8
    shi = np.rint(sh).astype(np.int64)
    rel = float(torch.linalg.norm(W - Q) / torch.linalg.norm(W))
    rec = dict(tensor=name, rows=f"{a}..{b}", m=m, n=n, full_shape=list(shape), ideal_bpw=ideal / N, physical_bpw=phys / N,
               diff_bpw=(phys - ideal) / N, steps_bpw=16 * m / N, fixed_bpw=8 * FIXED / N, prop_overhead_bpw=prop,
               escapes=int((shi >= 128).sum()), max_shell=int(shi.max()), rel_fro_err=rel, codes_valid=valid,
               ideal_hist_vs_codes_equal=abs(ideal - ideal2) < 1e-6, decode_max_abs_diff=maxdiff,
               roundtrip_codes=rt_codes, roundtrip_steps=rt_steps)
    out.append(rec)
    P(f"{name.split('.')[-3] if 'layers' in name else name} {a}:{b} m={m} n={n} ideal={rec['ideal_bpw']:.4f} phys={rec['physical_bpw']:.4f} "
      f"steps={rec['steps_bpw']:.5f} fixed={rec['fixed_bpw']:.5f} prop={prop:.5f} esc={rec['escapes']} maxshell={rec['max_shell']} "
      f"relerr={rel:.4f} valid={valid} maxdiff={maxdiff:.2e} rt_codes={rt_codes} rt_steps={rt_steps}")
mean_prop = float(np.mean([o['prop_overhead_bpw'] for o in out]))
allrt = all(o['roundtrip_codes'] and o['roundtrip_steps'] and o['codes_valid'] for o in out)
P(f"mean proportional overhead {mean_prop:.5f} bpw; all exact: {allrt}")
json.dump(dict(multiplier=MULT, cases=out, mean_prop_overhead_bpw=mean_prop, all_roundtrips_exact=allrt, printed_output=buf.getvalue()),
          open(os.path.join(HERE, '..', 'results', 'codec_real_weights.json'), 'w'), indent=1)
