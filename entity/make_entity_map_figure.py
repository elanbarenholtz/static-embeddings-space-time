"""Figure: held-out predicted locations for the entity-covered test places, by representation.

Reads entity_map_preds.csv (written by map_csv.py) and the Natural Earth 110m coastline
(ne_110m_coastline.zip, https://www.naturalearthdata.com/). Needs: numpy, pandas, matplotlib, pyshp.
Writes fig_entity_map_row.png (one row, with coastlines) and, with `2x2`, fig_entity_map.png (the figure in the paper:
two rows, no coastlines, R^2 for latitude / longitude in the titles). Points are colored by the continent of the item's labeled country.
"""
import io, sys, zipfile, tempfile, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = {'North America': '#D55E00', 'Europe': '#0072B2', 'Asia': '#E69F00',
          'Africa': '#009E73', 'South America': '#CC79A7', 'Oceania': '#56B4E9'}

SOUTH_AMERICA = 'Argentina Bolivia Brazil Chile Colombia Ecuador Falkland_Islands Guyana Paraguay Peru Suriname Uruguay Venezuela Los_Ríos_Region'
NORTH_AMERICA = ('Antigua_and_Barbuda Bahamas The_Bahamas Barbados Belize Bermuda British_Virgin_Islands Canada Caribbean Costa_Rica Cuba '
                 'Dominica Dominican_Republic El_Salvador Greenland Grenada Guatemala Haiti Honduras Jamaica Mexico Montserrat Nevis '
                 'Newfoundland_and_Labrador Nicaragua Panama Puerto_Rico Saint_Lucia Saint_Vincent_and_the_Grenadines Trinidad_and_Tobago '
                 'U.S._Virgin_Islands United_States United_States_of_America Crow_Nation Hopi Hudson_County')
AFRICA = ('Algeria Angola Benin Botswana Burkina_Faso Burundi Cameroon Cape_Verde Central_African_Republic Chad Comoros Côte_d\'Ivoire '
          'DR_Congo Democratic_Republic_of_the_Congo Egypt Equatorial_Guinea Eritrea Eswatini Ethiopia Ghana Guinea Guinea-Bissau Ivory_Coast '
          'Kenya Lesotho Liberia Libya Madagascar Malawi Mali Mauritania Mauritius Morocco Mozambique Namibia Niger Nigeria Otuke Puntland '
          'Republic_of_the_Congo Rwanda Senegal Seychelles Sierra_Leone Somalia Somaliland South_Africa South_Sudan Sudan Tanzania Togo Tunisia '
          'Uganda Western_Sahara Zambia Zimbabwe')
OCEANIA = ('Australia Cook_Islands Federated_States_of_Micronesia Kiribati Marshall_Islands Nauru New_Zealand Palau Papua_New_Guinea Samoa '
           'Solomon_Islands Tonga Tuvalu Vanuatu')
ASIA = ('Afghanistan Armenia Azerbaijan Bahrain Bangladesh Bhutan British_Indian_Ocean_Territory Brunei Burma Cambodia China Cyprus De_facto '
        'Dutch_East_Indies East_Timor Georgia_(country) Hong_Kong India Indonesia Iran Iraq Israel Japan Jordan Kazakhstan Kuwait Kyrgyzstan '
        'Laos Lebanon Malaysia Maldives Mandatory_Palestine Mongolia Myanmar Nepal North_Korea Oman Pakistan Palestinian_Territories '
        'People\'s_Republic_of_China Philippines Qatar Republic_of_China Saudi_Arabia Singapore South_Korea Sri_Lanka State_of_Palestine Syria '
        'Taiwan Tajikistan Thailand Turkey Turkmenistan United_Arab_Emirates Uzbekistan Vietnam Yemen')
CONT = {}
for name, group in [('South America', SOUTH_AMERICA), ('North America', NORTH_AMERICA), ('Africa', AFRICA), ('Oceania', OCEANIA), ('Asia', ASIA)]:
    for c in group.split(): CONT[c] = name


def continent(country):
    # everything not listed above is European (including UK counties and former states)
    # Antarctica has no color of its own and is omitted from the map
    if country == 'Antarctica': return None
    return CONT.get(country, 'Europe')


def coastlines():
    import shapefile  # pyshp (only needed for the row figure with coastlines)
    z = zipfile.ZipFile(os.path.join(HERE, 'ne_110m_coastline.zip')); d = tempfile.mkdtemp()
    z.extractall(d); shp = [f for f in os.listdir(d) if f.endswith('.shp')][0]
    r = shapefile.Reader(os.path.join(d, shp)); lines = []
    for s in r.shapes():
        pts = np.array(s.points); parts = list(s.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]): lines.append(pts[a:b])
    return lines


def main(out='fig_entity_map_row.png'):
    df = pd.read_csv(os.path.join(HERE, 'entity_map_preds.csv'))
    full = df.copy()   # R^2 uses every test place; the four Antarctic items are only left off the map
    df['cont'] = df['country'].map(continent); df = df[df['cont'].notna()].reset_index(drop=True)
    def r2(p, y): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    def ttl(name, a, b):
        return f'{name}  ($R^2$ = {r2(full[a], full.latitude):.2f} / {r2(full[b], full.longitude):.2f})'
    panels = [('True locations', 'latitude', 'longitude'),
              (ttl('Static, word average', 'w2v_words_lat', 'w2v_words_lon'), 'w2v_words_lat', 'w2v_words_lon'),
              (ttl('Static, entity', 'entity_lat', 'entity_lon'), 'entity_lat', 'entity_lon'),
              (ttl('Llama-2-7B', 'llama_lat', 'llama_lon'), 'llama_lat', 'llama_lon')]
    lines = coastlines()
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.05), sharey=True)
    for ax, (title, la, lo) in zip(axes, panels):
        for ln in lines: ax.plot(ln[:, 0], ln[:, 1], color='#9aa5b8', lw=0.5, zorder=1)
        for c, col in COLORS.items():
            m = df['cont'] == c
            ax.scatter(df.loc[m, lo], df.loc[m, la], s=3, c=col, alpha=0.55, linewidths=0, zorder=2)
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_title(title, fontsize=10.5, loc='left')
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_xlabel('longitude')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('latitude')
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, markersize=5, label=n) for n, c in COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(HERE, out), dpi=300); print('wrote', out, len(df), 'places')



def r2(y, p): return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main_2x2(out='fig_entity_map.png'):
    full = pd.read_csv(os.path.join(HERE, 'entity_map_preds.csv'))        # R^2 is computed on all places
    df = full.copy(); df['cont'] = df['country'].map(continent); df = df[df['cont'].notna()].reset_index(drop=True)  # Antarctica is not drawn
    panels = [('True locations', None, None), ('Wikipedia2Vec, word average', 'w2v_words_lat', 'w2v_words_lon'),
              ('Wikipedia2Vec, entity', 'entity_lat', 'entity_lon'), ('Llama-2-7B, layer 24', 'llama_lat', 'llama_lon')]
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True, sharey=True)
    for ax, (title, la, lo) in zip(axes.ravel(), panels):
        pla, plo = (df['latitude'], df['longitude']) if la is None else (df[la], df[lo])
        for c, col in COLORS.items():
            m = df['cont'] == c
            ax.scatter(plo[m].clip(-180, 180), pla[m].clip(-60, 80), s=3, c=col, alpha=0.55, linewidths=0, zorder=2)
        if la is not None:
            title += f':  $R^2$ = {r2(full["latitude"], full[la]):.2f} / {r2(full["longitude"], full[lo]):.2f}'
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_title(title, loc='left', fontsize=11)
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.grid(color='0.95', lw=0.5)
    for ax in axes[1]: ax.set_xlabel('longitude')
    for ax in axes[:, 0]: ax.set_ylabel('latitude')
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, markersize=5, label=n) for n, c in COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(HERE, out), dpi=300); print('wrote', out, len(df), 'places')


if __name__ == '__main__':
    if sys.argv[1:2] == ['2x2']: main_2x2(*sys.argv[2:])
    else: main(*sys.argv[1:])
