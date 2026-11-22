"""Chargeur du professeur Inkling 2 Mini (W4A16, déquantification Transformers) — formic_inkling."""
import os, time, torch
MODEL_DIR = os.environ.get("INK_MODEL_DIR", "/root/models/inkling-2-mini")

def load_teacher(dtype=torch.bfloat16, experts_impl="grouped_mm", device="cuda"):
    from transformers import AutoModelForImageTextToText, AutoConfig
    t = time.time()
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    try:
        cfg.quantization_config = None
    except Exception:
        pass
    from transformers import Inkling2LoopForConditionalGeneration as C
    m = C.from_pretrained(MODEL_DIR, dtype=dtype, device_map=device, experts_implementation=experts_impl)
    m.eval()
    print(f"[ink_model] loaded in {time.time()-t:.0f}s", flush=True)
    return m

def text_model(m):
    return m.model.language_model

def head(m):
    return m.lm_head
