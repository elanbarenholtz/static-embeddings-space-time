"""Figure: words whose similarity to a place name tracks the place's latitude or longitude (released places).
Reads word_coords_released.json (written by word_coords.py); writes fig_word_coords.png."""
import json, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); N = 12
d = json.load(open(os.path.join(HERE, 'word_coords_released.json')))
plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white'})
panels = [('latitude', 'top_neg', 'Words closer to more southern places', '#D55E00', -1),
          ('latitude', 'top_pos', 'Words closer to more northern places', '#0072B2', +1),
          ('longitude', 'top_neg', 'Words closer to more western places', '#009E73', -1),
          ('longitude', 'top_pos', 'Words closer to more eastern places', '#CC79A7', +1)]
fig, axes = plt.subplots(2, 2, figsize=(10, 5.5))
for ax, (coord, key, title, col, sign) in zip(axes.ravel(), panels):
    items = d[coord][key][:N]; words = [w for w, _ in items]; vals = [abs(v) for _, v in items]
    y = list(range(N))[::-1]
    ax.barh(y, vals, color=col, height=0.7)
    ax.set_yticks(y); ax.set_yticklabels(words, fontsize=11.5)
    for yi, v, (_, raw) in zip(y, vals, items): ax.text(v + 0.006, yi, f'{raw:+.2f}', va='center', fontsize=9, color='#333')
    ax.set_xlim(0, 0.62); ax.set_title(title, loc='left', fontsize=11.5)
    ax.set_xlabel(f'{"" if sign > 0 else chr(8722)}$r$ with {coord}')
fig.tight_layout(h_pad=1.6, w_pad=2.2); fig.savefig(os.path.join(HERE, 'fig_word_coords.png'), dpi=300); print('ok')
