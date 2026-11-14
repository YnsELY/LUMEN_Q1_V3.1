import importlib, harness, seq, formic_q, time
importlib.reload(formic_q); _S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
importlib.reload(seq); SQ = seq
H.restore(); t = time.time(); print("big teacher cache:", H.build_teacher_big(), "%.0fs" % (time.time() - t), flush=True)
