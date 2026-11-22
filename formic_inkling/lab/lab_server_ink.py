"""Serveur de laboratoire FORMIC-Ink (dérivé de formic_arch/lab/lab_server.py) : charge le professeur Inkling 2 Mini une fois
(chargeur ink_model.load_teacher, W4A16, experts déquantifiés par Transformers) et exécute les jobs déposés dans la file.
Usage : python -B lab_server_ink.py   (jobs : /workspace/work_ink/queue/NNN_nom.py -> logs/NNN_nom.log)"""
import os, sys, time, glob, traceback, io, contextlib
import torch
MODEL_DIR = "/root/models/inkling-2-mini"
QUEUE = "/workspace/work_ink/queue"; LOGS = "/workspace/work_ink/logs"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
t0 = time.time()
from transformers import AutoTokenizer
import ink_model
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = ink_model.load_teacher()
for p in model.parameters(): p.requires_grad_(False)
print("loaded in %.1fs" % (time.time() - t0), flush=True)
NS = {"model": model, "tok": tok, "torch": torch, "MODEL_DIR": MODEL_DIR}

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
