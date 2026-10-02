import os as _os_sb
_SBROOT = _os_sb.environ.get('SB_ROOT', _os_sb.path.dirname(_os_sb.path.dirname(_os_sb.path.abspath(__file__))))
import json, glob, os, csv
from emb import build_cache
R = os.path.dirname(os.path.abspath(__file__))
texts = []
D = json.load(open(f'{R}/painrepo/datasets/3.1_pain_and_control_datasets.json'))['datasets']
for k, v in D.items():
    texts += [s['prompt'] for s in v['sentences']]
S = json.load(open(f'{R}/painrepo/datasets/3.1_sadness_dataset.json'))['datasets']
for k, v in S.items():
    texts += [s['prompt'] for s in v['sentences']]
for f in glob.glob(_os_sb.path.join(_SBROOT, 'review_v2', 'data', 'qwen-emotion-stories', 'corpus', '*.jsonl')):
    for l in open(f):
        d = json.loads(l); texts.append(d['text']); texts.append(d['emotion'])
for s in ['train', 'dev', 'test']:
    for row in csv.reader(open(f'{R}/data/{s}.tsv'), delimiter='\t'):
        texts.append(row[0])
texts.append(open(f'{R}/data/emotions.txt').read())
texts.append('afraid angry ashamed calm desperate disgusted excited joyful lonely proud sad surprised neutral')
print(len(texts), 'texts', flush=True)
build_cache('review', texts)
