"""96 invites image + texte (4 domaines x 24 : documents DocVQA, graphiques ChartQA, captures d'écran ScreenSpot, photos COCO),
images réduites à 640 px de grand côté, entrées du processeur d'Inkling sauvegardées. Sortie : /workspace/work_ink/data/vision/vision_inputs.pt"""
import os, torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from datasets import load_dataset
from transformers import AutoProcessor
P = AutoProcessor.from_pretrained('/root/models/inkling-2-mini')
SRC = [('documents', ('lmms-lab/DocVQA', 'DocVQA'), 'validation', lambda r: r['question']),
       ('charts', ('HuggingFaceM4/ChartQA',), 'test', lambda r: r['query']),
       ('screenshots', ('rootsautomation/ScreenSpot',), 'test', lambda r: 'Describe what is shown on this screen, then explain how to: ' + r['instruction']),
       ('photos', ('detection-datasets/coco',), 'val', lambda r: 'Describe this photo in detail.')]
out = []
for dom, a, sp, q in SRC:
    n = 0
    for j, r in enumerate(load_dataset(*a, split=sp, streaming=True)):
        if n >= 24: break
        if dom == 'screenshots' and j % 5: continue
        im = r['image'].convert('RGB'); s = 640 / max(im.size)
        if s < 1: im = im.resize((max(28, int(im.size[0] * s)), max(28, int(im.size[1] * s))))
        msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': q(r)}]}]
        t = P.tokenizer.apply_chat_template(msgs, tokenize=False, reasoning_effort='none') + '<|message_model|>'
        x = P(text=[t], images=[im], return_tensors='pt')
        out.append(dict(domain=dom, source=f'{a[0]}:{sp}:{j}', **{k: v for k, v in x.items()})); n += 1
    print(dom, n, flush=True)
torch.save(out, '/workspace/work_ink/data/vision/vision_inputs.pt')
print('saved', len(out))
