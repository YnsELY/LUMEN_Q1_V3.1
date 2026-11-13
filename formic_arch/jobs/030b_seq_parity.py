import importlib, seq
importlib.reload(seq); SQ = seq
H.restore()
print("Seq layer-by-layer vs model forward (BF16, 16 docs):", SQ.seq_parity(16))
print("alloc GiB %.1f" % (torch.cuda.memory_allocated()/2**30))
