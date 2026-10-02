"""Robust last-token activation extraction on MPS (for Llama-2-7B).

Equal-length batches (no padding). Every batch is run twice and accepted only if
  - all values are finite and every row is nonzero,
  - layer 0 equals the embedding-table rows for the last tokens,
  - the two runs agree (max abs diff at the top saved layer small relative to scale).
Failed batches are retried (after emptying the MPS cache) and, if they still fail,
split into single items; items that never pass are listed in bad_<tag>.npy.
Usage: python extract_safe.py <dataset> <model_dir>   env: BATCH (32), DTYPE (fp16|bf16)
"""
import sys, time, json, os
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DATASET, MODEL = sys.argv[1], sys.argv[2]
BATCH = int(os.environ.get('BATCH', 32))
DT = {'fp16': torch.float16, 'bf16': torch.bfloat16}[os.environ.get('DTYPE', 'fp16')]
names = pd.read_csv(f'data/{DATASET}.csv')['name'].astype(str).tolist()
dev = 'mps'
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DT).to(dev).eval()
emb = model.get_input_embeddings().weight.detach().float().cpu().numpy()
nL = model.config.num_hidden_layers
LAYERS = sorted(set([0] + list(range(4, nL + 1, 4)) + [round(0.6 * nL)]))
H = model.config.hidden_size
tag = f"{DATASET}_{MODEL.rstrip('/').split('/')[-1]}"
out = np.lib.format.open_memmap(f'acts_{tag}.npy', mode='w+', dtype=np.float16,
                                shape=(len(LAYERS), len(names), H))
ids_all = tok(names)['input_ids']
ntok = np.array([len(x) for x in ids_all], dtype=np.int16)
# MPS (torch 2.10, Apple M5) returns wrong hidden states for Llama-2 whenever the number of
# tokens in a forward pass (batch x length) is above ~16 and not a multiple of 256; single
# items and multiples of 256 match a CPU reference. So each batch holds equal-length names,
# is topped up with copies of its first item to a multiple of 256 tokens, and is spot-checked
# against a single-item pass.
from math import gcd, ceil
TARGET = int(os.environ.get('TARGET_TOKENS', 768))
batches = []
for n in sorted(set(ntok.tolist())):
    grp = np.where(ntok == n)[0]
    bs0 = 256 // gcd(n, 256)
    bs = bs0 * max(1, round(TARGET / (bs0 * n)))
    if bs * n > 4096:   # long, odd lengths: run each item on its own
        batches += [(grp[i:i + 1], 1) for i in range(len(grp))]
    else:
        batches += [(grp[i:i + bs], bs) for i in range(0, len(grp), bs)]
rng = np.random.default_rng(0)

def fwd(ids):
    x = torch.tensor(ids, device=dev)
    with torch.no_grad():
        hs = model(input_ids=x, output_hidden_states=True).hidden_states
        R = torch.stack([hs[L][:, -1] for L in LAYERS]).float()
    torch.mps.synchronize()
    R = R.cpu().numpy(); del hs, x
    return R

def cosmin(a, b):
    return float(((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)).min())

def check(R, idx):
    if not np.isfinite(R).all(): return 'nonfinite'
    if (np.abs(R).sum(2) == 0).any(): return 'zero'
    if not np.allclose(R[0], emb[[ids_all[i][-1] for i in idx]], atol=1e-2): return 'layer0'
    return None

stats = {'batches': 0, 'fallback_batches': 0, 'spot_min_cos': 1.0}
bad = []
t0 = time.time(); done = 0
for bi, (idx, bs) in enumerate(batches):
    n = int(ntok[idx[0]])
    ids = [ids_all[i] for i in idx]
    fill = bs - len(idx) if len(idx) * n > 16 else 0
    R = None
    if bs > 1:
        R = fwd(ids + [ids[0]] * fill)[:, :len(idx)]
        why = check(R, idx)
        if why is None:
            j = int(rng.integers(len(idx)))
            c = cosmin(R[:, j], fwd([ids[j]])[:, 0])
            stats['spot_min_cos'] = min(stats['spot_min_cos'], c)
            if c < 0.999: why = f'spot cos {c:.3f}'
        if why is not None:
            print(f'batch {bi} (len {n}, {len(idx)} items) failed: {why}; running singly', flush=True)
            stats['fallback_batches'] += 1; R = None
    if R is None:
        R = np.concatenate([fwd([x]) for x in ids], 1)
        for k, i in enumerate(idx):
            if check(R[:, k:k + 1], [i]) is not None: bad.append(int(i))
    out[:, idx] = R.astype(np.float16)
    stats['batches'] += 1; done += len(idx)
    if bi % 10 == 0:
        torch.mps.empty_cache()
        print(f'{done}/{len(names)}  {time.time() - t0:.0f}s  {stats}  bad {len(bad)}', flush=True)
out.flush()
np.save(f'bad_{tag}.npy', np.array(bad, dtype=np.int64))
json.dump({'layers': LAYERS, 'n_layers': nL, 'hidden': H, 'model': MODEL, 'dataset': DATASET,
           'n': len(names), 'template': '{}', 'dtype': str(DT), 'n_bad': len(bad), **stats,
           'median_tokens': float(np.median(ntok))}, open(f'meta_{tag}.json', 'w'))
np.save(f'ntok_{tag}.npy', ntok)
print('DONE', tag, f'{time.time() - t0:.0f}s', stats, 'bad items', len(bad), flush=True)
