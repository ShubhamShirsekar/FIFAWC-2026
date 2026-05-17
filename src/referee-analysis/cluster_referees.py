import json
import os
import numpy as np
import pandas as pd
import requests
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
REFEREES_DIR = '../data/referees'
OUTPUT_DIR   = './output'
OLLAMA_URL   = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'gemma4:e4b-q4_k_m'
FORCED_K     = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── STEP 1: LOAD & AGGREGATE FEATURES PER REFEREE ───────────────────────────
def load_referee(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def aggregate_features(data, referee_name):
    total_fixtures     = 0
    weighted_cards_sum = 0
    weighted_fouls_sum = 0
    fouls_fixtures     = 0
    card_avgs          = []
    competitions       = set()

    for entry in data:
        fixtures  = entry.get('fixtures') or 0
        if fixtures == 0:
            continue

        cards_avg = entry.get('cards', {}).get('avgPerGame')
        fouls_avg = entry.get('fouls', {}).get('avgPerGame')
        comp      = entry.get('competition', '')

        if cards_avg is not None:
            weighted_cards_sum += cards_avg * fixtures
            card_avgs.append(cards_avg)

        if fouls_avg is not None:
            weighted_fouls_sum += fouls_avg * fixtures
            fouls_fixtures     += fixtures

        total_fixtures += fixtures
        competitions.add(comp)

    if total_fixtures == 0:
        return None

    weighted_avg_cards = weighted_cards_sum / total_fixtures if total_fixtures > 0 else 0
    weighted_avg_fouls = weighted_fouls_sum / fouls_fixtures if fouls_fixtures > 0 else None
    card_consistency   = float(np.std(card_avgs)) if len(card_avgs) > 1 else 0.0

    return {
        'referee':            referee_name,
        'total_fixtures':     total_fixtures,
        'competition_count':  len(competitions),
        'weighted_avg_cards': round(weighted_avg_cards, 3),
        'card_consistency':   round(card_consistency, 3),
        'weighted_avg_fouls': round(weighted_avg_fouls, 3) if weighted_avg_fouls else None,
    }

# ─── LOAD ALL REFEREES ────────────────────────────────────────────────────────
print('Loading referee data...')
records = []

for filename in os.listdir(REFEREES_DIR):
    if not filename.endswith('.json'):
        continue
    referee_name = filename.replace('.json', '').replace('_', ' ')
    filepath     = os.path.join(REFEREES_DIR, filename)
    data         = load_referee(filepath)
    features     = aggregate_features(data, referee_name)
    if features:
        records.append(features)

df = pd.DataFrame(records)
print(f'Loaded {len(df)} referees successfully')
print(f'\nFeature summary:')
print(df.describe())

# ─── STEP 2: PREPARE FEATURES FOR CLUSTERING ─────────────────────────────────
# Derived feature: card trigger threshold
df['cards_per_100_fouls'] = (
    (df['weighted_avg_cards'] / df['weighted_avg_fouls']) * 100
).round(3)

# Check fouls availability
fouls_available = df['weighted_avg_fouls'].notna().sum()
use_fouls       = fouls_available >= (len(df) * 0.3)

if use_fouls:
    df['weighted_avg_fouls'] = df['weighted_avg_fouls'].fillna(
        df['weighted_avg_fouls'].median()
    )
    print(f'\nIncluding fouls data ({fouls_available} referees have it)')
else:
    print(f'\nSkipping fouls data (only {fouls_available} referees have it)')

# Purely behavioral features
feature_cols = [
    'weighted_avg_cards',
    'weighted_avg_fouls',
    'card_consistency',
    'cards_per_100_fouls'
]

print('\nBehavioral feature summary:')
print(df[feature_cols].describe())

X        = df[feature_cols].fillna(0).values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── STEP 3: FIND OPTIMAL CLUSTER COUNT ──────────────────────────────────────
print('\nFinding optimal cluster count...')
inertias    = []
silhouettes = []
k_range     = range(2, 9)

for k in k_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, lbl)
    silhouettes.append(sil)
    print(f'  k={k} | Inertia: {km.inertia_:.1f} | Silhouette: {sil:.3f}')

# Plot elbow + silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(list(k_range), inertias, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax1.grid(True)

ax2.plot(list(k_range), silhouettes, 'go-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score (higher = better)')
ax2.set_title('Silhouette Scores')
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'elbow_silhouette.png'), dpi=150)
print(f'\nElbow plot saved to output/elbow_silhouette.png')

# ─── STEP 4: APPLY K-MEANS ───────────────────────────────────────────────────
print(f'\nApplying K-Means with k={FORCED_K}...')

km_final      = KMeans(n_clusters=FORCED_K, random_state=42, n_init=10)
df['cluster'] = km_final.fit_predict(X_scaled)

# ─── STEP 5: COMPUTE CLUSTER STATS ───────────────────────────────────────────
cluster_stats = df.groupby('cluster').agg(
    avg_cards   = ('weighted_avg_cards',  'mean'),
    avg_fouls   = ('weighted_avg_fouls',  'mean'),
    avg_trigger = ('cards_per_100_fouls', 'mean'),
    avg_consist = ('card_consistency',    'mean'),
    count       = ('referee',             'count')
).round(2)

print('\nCluster stats:')
print(cluster_stats)

for c in sorted(df['cluster'].unique()):
    group = df[df['cluster'] == c]
    print(f'\nCluster {c} ({len(group)} referees):')
    print(f'  Avg cards/game:     {cluster_stats.loc[c, "avg_cards"]}')
    print(f'  Avg fouls/game:     {cluster_stats.loc[c, "avg_fouls"]}')
    print(f'  Trigger threshold:  {cluster_stats.loc[c, "avg_trigger"]}')
    print(f'  Consistency (std):  {cluster_stats.loc[c, "avg_consist"]}')
    print(f'  Referees: {", ".join(sorted(group["referee"].tolist()))}')

# ─── STEP 6: PCA VISUALIZATION ───────────────────────────────────────────────
pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
plt.figure(figsize=(14, 9))

for c in sorted(df['cluster'].unique()):
    mask = df['cluster'] == c
    idxs = df[mask].index.tolist()

    plt.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=colors[c % len(colors)],
        label=f'Cluster {c}',
        s=120, alpha=0.85,
        edgecolors='white', linewidth=0.5
    )

    for idx in idxs:
        pos = df.index.tolist().index(idx)
        plt.annotate(
            df.loc[idx, 'referee'].replace('-', ' '),
            (X_pca[pos, 0], X_pca[pos, 1]),
            fontsize=6, ha='center', va='bottom', alpha=0.8
        )

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('WC 2026 Referee Behavioral Clusters', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'referee_clusters_pca.png'), dpi=150)
print('\nPCA plot saved to output/referee_clusters_pca.png')

# ─── STEP 7: GENERATE DESCRIPTIONS VIA OLLAMA ────────────────────────────────
def get_cluster_description(cluster_id, stats, referees):
    prompt = f"""You are a witty football pundit. In one punchy, funny sentence describe a group of {stats['count']} World Cup 2026 referees who average {stats['avg_cards']} cards/game, {stats['avg_fouls']} fouls/game, and book a player every {round(100/stats['avg_trigger'], 1)} fouls. Be creative, humorous and specific to the numbers."""

    try:
        response = requests.post(
            'http://localhost:11434/api/chat',
            json={
                'model':  OLLAMA_MODEL,
                'stream': False,
                'messages': [
                    {
                        'role':    'user',
                        'content': prompt
                    }
                ]
            },
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            text = data.get('message', {}).get('content', '').strip()
            return text if text else 'No description generated'
        else:
            return f'Ollama error: HTTP {response.status_code}'

    except Exception as e:
        return f'Ollama error: {str(e)}'


print('\nGenerating cluster descriptions via Ollama...\n')

cluster_profiles = []
desc_map         = {}

for c in sorted(df['cluster'].unique()):
    group     = df[df['cluster'] == c]
    stats_row = cluster_stats.loc[c].to_dict()
    referees  = sorted(group['referee'].str.replace('-', ' ').tolist())

    print(f'Generating description for Cluster {c} ({len(group)} referees)...')
    description  = get_cluster_description(c, stats_row, referees)
    desc_map[c]  = description

    print(f'\nCluster {c} Description:')
    print(description)
    print('-' * 60)

    cluster_profiles.append({
        'cluster_id':  int(c),
        'size':        int(stats_row['count']),
        'avg_cards':   stats_row['avg_cards'],
        'avg_fouls':   stats_row['avg_fouls'],
        'avg_trigger': stats_row['avg_trigger'],
        'avg_consist': stats_row['avg_consist'],
        'referees':    referees,
        'description': description
    })

# ─── STEP 8: SAVE ALL RESULTS ─────────────────────────────────────────────────
# Cluster profiles
with open(os.path.join(OUTPUT_DIR, 'cluster_profiles.json'), 'w') as f:
    json.dump(cluster_profiles, f, indent=2)

# Individual referee records
output = []
for _, row in df.iterrows():
    output.append({
        'referee':             row['referee'],
        'cluster_id':          int(row['cluster']),
        'cluster_description': desc_map[int(row['cluster'])],
        'avg_cards_per_game':  row['weighted_avg_cards'],
        'avg_fouls_per_game':  row['weighted_avg_fouls'],
        'card_consistency':    row['card_consistency'],
        'cards_per_100_fouls': row['cards_per_100_fouls'],
        'total_fixtures':      int(row['total_fixtures']),
    })

with open(os.path.join(OUTPUT_DIR, 'referee_clusters.json'), 'w') as f:
    json.dump(output, f, indent=2)

print('\n✅ Done. Files saved to output/')
print('   - referee_clusters.json   (individual referee assignments)')
print('   - cluster_profiles.json   (cluster summaries + descriptions)')
print('   - referee_clusters_pca.png')
print('   - elbow_silhouette.png')