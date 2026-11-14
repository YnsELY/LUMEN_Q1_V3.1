"""Persistent experiment server: loads the BF16 model once on the GPU and executes job
scripts dropped in a queue directory, in a shared namespace (model stays in VRAM).
Usage: python lab_server.py  (jobs: /workspace/work/queue/NNN_name.py -> logs/NNN_name.log)
"""
import os, sys, time, glob, traceback, io, contextlib, json, shutil
import torch
MODEL_DIR = "/root/models/qwen3.8-27b"
QUEUE = "/workspace/work/queue"; LOGS = "/workspace/work/logs"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
t0 = time.time()
import transformers
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model, loading_info = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True, local_files_only=True)
model.eval()
for p in model.parameters(): p.requires_grad_(False)
print("loaded in %.1fs" % (time.time() - t0), flush=True)
print("loading_info:", {k: (v if len(v) < 40 else v[:40] + ['...%d' % len(v)]) for k, v in loading_info.items()}, flush=True)
NS = {"model": model, "tok": tok, "torch": torch, "MODEL_DIR": MODEL_DIR, "loading_info": loading_info}

class Tee(io.TextIOBase):
    def __init__(self, f): self.f = f
    def write(self, s):
        self.f.write(s); self.f.flush(); return len(s)

while True:
    jobs = sorted(glob.glob(os.path.join(QUEUE, "*.py")))
    if not jobs:
        if os.path.exists(os.path.join(QUEUE, "STOP")): break
        time.sleep(1.0); continue
    job = jobs[0]; name = os.path.basename(job)[:-3]
    log = os.path.join(LOGS, name + ".log")
    src = open(job).read()
    os.rename(job, os.path.join(LOGS, name + ".py.ran"))
    with open(log, "w") as f, contextlib.redirect_stdout(Tee(f)), contextlib.redirect_stderr(Tee(f)):
        ts = time.time()
        try:
            exec(compile(src, job, "exec"), NS)
            status = "OK"
        except BaseException as e:
            traceback.print_exc(); status = "ERROR"
            if isinstance(e, KeyboardInterrupt): raise
        print("\n[job %s %s in %.1fs]" % (name, status, time.time() - ts))
    open(os.path.join(LOGS, name + ".done"), "w").write(status)
    torch.cuda.empty_cache()
