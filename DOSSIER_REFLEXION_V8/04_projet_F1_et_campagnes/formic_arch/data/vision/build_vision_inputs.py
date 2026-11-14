import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, hashlib, textwrap, random
import torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from transformers import AutoProcessor

OUT = "/workspace/work/data/vision"; IMG = OUT + "/images"; os.makedirs(IMG, exist_ok=True)
MAXS = 448
FD = "/usr/share/fonts/truetype/dejavu/"
LIC = {4: "CC BY 2.0", 5: "CC BY-SA 2.0", 7: "No known copyright restrictions (Flickr Commons)", 8: "United States Government Work"}

def fit(im):
    im = im.convert("RGB"); s = MAXS / max(im.size)
    if s < 1: im = im.resize((max(1, round(im.width*s)), max(1, round(im.height*s))), Image.LANCZOS)
    return im

samples = []  # (id, kind, source, license, instruction, PIL)
# photos
ds = load_dataset("lmms-lab/COCO-Caption2017", split="val", streaming=True)
n = 0
for i, x in enumerate(ds):
    if i < 500: continue
    if x["license"] not in LIC: continue
    samples.append((f"photo_{n:02d}", "photo", f"lmms-lab/COCO-Caption2017 (COCO val2017) {x['file_name']} {x['coco_url']}",
                    f"COCO per-image license id {x['license']}: {LIC[x['license']]}; dataset card: CC BY 4.0 annotations", "Describe this image in detail.", fit(x["image"])))
    n += 1
    if n == 12: break
assert n == 12

# text pages
paras = [
 "The lighthouse keeper woke before dawn each morning to trim the wick and polish the great lens. Ships passing in the night depended on that steady beam, and he took the duty seriously, even when storms rattled the windows and the sea climbed the rocks below.",
 "A good loaf of bread requires only four ingredients: flour, water, salt and yeast. The difference between an average loaf and an excellent one lies in patience. Let the dough rest overnight in a cool place, and the flavour will deepen remarkably.",
 "Public libraries began as private collections shared among friends. Over the centuries they grew into civic institutions, offering quiet rooms, free access to books and, more recently, computers, classes and community events for people of every age.",
 "Rivers shape the land slowly. Water carries sediment downstream, carving valleys and depositing fertile soil on the plains. Cities have long grown along their banks, drawn by easy transport, drinking water and rich farmland nearby.",
]
for k, (p, fs) in enumerate(zip(paras, [14, 18, 22, 16])):
    W = 448 if False else 448
    f = ImageFont.truetype(FD + ("DejaVuSans.ttf" if k % 2 == 0 else "DejaVuSerif.ttf"), fs)
    cw = f.getlength("n")
    lines = textwrap.wrap(p, width=max(10, int((W-40)/cw*1.0)))
    lh = int(fs*1.4); H = 40 + lh*len(lines)
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
    for j, l in enumerate(lines): d.text((20, 20+j*lh), l, fill="black", font=f)
    samples.append((f"text_{k:02d}", "text", "generated for this project (PIL)", "generated for this project", "Transcribe the text in this image.", im))

def mpl_img(fig, path=None):
    import io
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=100); plt.close(fig); b.seek(0)
    return Image.open(b).convert("RGB")

# tables
tabs = [
 (["Month", "Sales", "Costs", "Profit"], [["Jan", 120, 80, 40], ["Feb", 135, 85, 50], ["Mar", 150, 90, 60], ["Apr", 110, 95, 15]]),
 (["Student", "Math", "Physics", "Art"], [["Ana", 91, 85, 72], ["Ben", 78, 82, 90], ["Cleo", 66, 70, 88], ["Dev", 99, 95, 61]]),
 (["City", "Pop (k)", "Area", "Rank"], [["Oslo", 709, 454, 1], ["Bergen", 286, 465, 2], ["Tromso", 77, 2521, 3], ["Alta", 21, 3849, 4]]),
 (["Item", "Qty", "Price", "Total"], [["Pen", 10, 1.5, 15.0], ["Book", 3, 12.0, 36.0], ["Lamp", 1, 25.5, 25.5], ["Bag", 2, 30.0, 60.0]]),
]
for k, (h, rows) in enumerate(tabs):
    f = ImageFont.truetype(FD + "DejaVuSans.ttf", 18); fb = ImageFont.truetype(FD + "DejaVuSans-Bold.ttf", 18)
    cw, rh = 100, 40; im = Image.new("RGB", (cw*4+20, rh*5+20), "white"); d = ImageDraw.Draw(im)
    for r, row in enumerate([h] + rows):
        for c, v in enumerate(row):
            x0, y0 = 10+c*cw, 10+r*rh
            if r == 0: d.rectangle([x0, y0, x0+cw, y0+rh], fill=(220, 230, 245))
            d.rectangle([x0, y0, x0+cw, y0+rh], outline="black")
            d.text((x0+8, y0+10), str(v), fill="black", font=fb if r == 0 else f)
    samples.append((f"table_{k:02d}", "table", "generated for this project (PIL)", "generated for this project", "Read this table and list its rows.", im))

# charts
cats = ["A", "B", "C", "D", "E"]
for k in range(4):
    fig, ax = plt.subplots(figsize=(4.48, 3.4))
    if k == 0:
        ax.bar(cats, [5, 9, 3, 7, 6], color="tab:blue", label="2023"); ax.bar(cats, [4, 6, 5, 8, 2], color="tab:orange", alpha=.7, label="2024")
        ax.set_title("Quarterly units sold"); ax.set_xlabel("Product"); ax.set_ylabel("Units (thousands)")
    elif k == 1:
        x = list(range(1, 9)); ax.plot(x, [2, 3, 5, 4, 6, 8, 7, 9], "o-", label="Site A"); ax.plot(x, [1, 2, 2, 3, 5, 5, 6, 6], "s--", label="Site B")
        ax.set_title("Weekly visitors"); ax.set_xlabel("Week"); ax.set_ylabel("Visitors (thousands)")
    elif k == 2:
        ax.pie([40, 25, 20, 15], labels=["Rent", "Food", "Travel", "Other"], autopct="%1.0f%%"); ax.set_title("Monthly budget split")
        ax.legend(["Rent", "Food", "Travel", "Other"], loc="lower right", fontsize=7)
    else:
        rnd = random.Random(1); xs = [rnd.uniform(0, 10) for _ in range(30)]
        ax.scatter(xs, [2*v+rnd.gauss(0, 2) for v in xs], label="Group 1"); ax.scatter(xs[:15], [20-v+rnd.gauss(0, 2) for v in xs[:15]], label="Group 2")
        ax.set_title("Height vs. weight"); ax.set_xlabel("Height index"); ax.set_ylabel("Weight index")
    if k != 2: ax.legend()
    fig.tight_layout()
    samples.append((f"chart_{k:02d}", "chart", "generated for this project (matplotlib)", "generated for this project", "Describe this chart, including its title, axes and the values shown.", mpl_img(fig)))

proc = AutoProcessor.from_pretrained("/root/models/qwen3.8-27b")
out, man = [], []
for sid, kind, src, lic, ins, im in samples:
    im = fit(im); im.save(f"{IMG}/{sid}.png")
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": ins}]}]
    r = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inp = {k: v.cpu() for k, v in r.items() if torch.is_tensor(v)}
    nimg = int((inp["input_ids"] == 248056).sum())
    assert nimg > 0 and "pixel_values" in inp and "image_grid_thw" in inp
    out.append({"id": sid, "kind": kind, "source": src, "license": lic, "instruction": ins, "inputs": inp})
    man.append({"id": sid, "kind": kind, "source": src, "license": lic, "image_size_wh": list(im.size), "n_image_tokens": nimg})
torch.save(out, f"{OUT}/vision_inputs.pt")
h = hashlib.sha256(open(f"{OUT}/vision_inputs.pt", "rb").read()).hexdigest()
json.dump({"processor": type(proc).__name__, "photo_dataset": "lmms-lab/COCO-Caption2017 (COCO val2017), skipped first 500 items, kept per-image licenses in {CC BY 2.0, CC BY-SA 2.0, no-known-restrictions, US Gov}",
           "sha256_vision_inputs_pt": h, "samples": man}, open(f"{OUT}/manifest.json", "w"), indent=1)
print("processor", type(proc).__name__)
