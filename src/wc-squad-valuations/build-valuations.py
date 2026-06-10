import argparse
import json
import re
import pandas as pd
import os
from pathlib import Path
import subprocess
import sys
import unicodedata
from rapidfuzz import fuzz

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = SCRIPT_DIR / 'output'
TRANSFERMARKT_DIR = REPO_ROOT / 'data' / 'Transfermarkt'
DEFAULT_PLAYERS_CSV = TRANSFERMARKT_DIR / 'players.csv'
LATEST_PLAYERS_DIR = TRANSFERMARKT_DIR / 'kaggle-latest'
LATEST_PLAYERS_CSV = LATEST_PLAYERS_DIR / 'players.csv'
KAGGLE_DATASET = 'davidcariboo/player-scores'
TOP_BOTTOM_COUNT = 3
os.makedirs(OUTPUT_DIR / 'countries', exist_ok=True)

# ─── COUNTRY NAME ALIASES: Wikipedia → players.csv country_of_citizenship ─────
COUNTRY_ALIASES = {
    'South Korea':            ['South Korea', 'Korea, South'],
    'USA':                    ['USA', 'United States', 'United States of America'],
    'Bosnia and Herzegovina': ['Bosnia and Herzegovina', 'Bosnia-Herzegovina'],
    'Ivory Coast':            ['Ivory Coast', "Cote d'Ivoire", "Côte d'Ivoire"],
    'DR Congo':               ['DR Congo', 'Congo DR'],
    'Trinidad and Tobago':    ['Trinidad and Tobago', 'Trinidad & Tobago'],
    'Turkey':                 ['Turkey', 'Türkiye', 'Turkiye'],
}


def normalize_text(value):
    value = unicodedata.normalize('NFKD', str(value))
    value = ''.join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace('&', 'and')
    value = re.sub(r'[^a-z0-9]+', ' ', value)
    return ' '.join(value.split())


def write_csv(df, path):
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback_path = path.with_name(f'{path.stem}_latest{path.suffix}')
        df.to_csv(fallback_path, index=False)
        print(f'  ⚠️  Could not overwrite {path.name}; wrote {fallback_path.name} instead')
        return fallback_path


def parse_args():
    parser = argparse.ArgumentParser(description='Build WC 2026 squad valuations from player market values.')
    parser.add_argument(
        '--refresh-players',
        action='store_true',
        help='Download the latest players.csv from Kaggle before running the analysis.',
    )
    parser.add_argument(
        '--players-csv',
        type=Path,
        help='Use a specific players.csv file instead of the default or Kaggle-downloaded snapshot.',
    )
    return parser.parse_args()


def download_latest_players():
    LATEST_PLAYERS_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        '-m',
        'kaggle',
        'datasets',
        'download',
        KAGGLE_DATASET,
        '--file',
        'players.csv',
        '--path',
        str(LATEST_PLAYERS_DIR),
        '--unzip',
        '--force',
    ]

    print('Refreshing players.csv from Kaggle...')
    subprocess.run(command, check=True)

    if not LATEST_PLAYERS_CSV.exists():
        raise FileNotFoundError(f'Kaggle download completed but {LATEST_PLAYERS_CSV} was not created')

    return LATEST_PLAYERS_CSV


def resolve_players_csv(args):
    if args.players_csv:
        return args.players_csv.resolve()

    if args.refresh_players:
        return download_latest_players()

    if LATEST_PLAYERS_CSV.exists():
        return LATEST_PLAYERS_CSV

    return DEFAULT_PLAYERS_CSV


# ─── LOAD ─────────────────────────────────────────────────────────────────────
args = parse_args()
players_csv_path = resolve_players_csv(args)

print('Loading files...')
with open(OUTPUT_DIR / 'wc2026_squads.json', encoding='utf-8') as squads_file:
    squads = json.load(squads_file)

players_df = pd.read_csv(players_csv_path)

# Clean DOB in players.csv → YYYY-MM-DD
players_df['dob'] = pd.to_datetime(
    players_df['date_of_birth'], errors='coerce'
).dt.strftime('%Y-%m-%d')
players_df['citizenship_norm'] = players_df['country_of_citizenship'].fillna('').apply(normalize_text)

print(f'Squads loaded:  {len(squads)} countries')
print(f'Players loaded: {len(players_df)} records')
print(f'Players source: {players_csv_path}')

# ─── EXTRACT DOB FROM WIKIPEDIA FORMAT ───────────────────────────────────────
def extract_dob(raw):
    m = re.search(r'\((\d{4}-\d{2}-\d{2})\)', str(raw))
    return m.group(1) if m else None

# ─── MATCH ONE PLAYER ─────────────────────────────────────────────────────────
def find_player(wiki_name, wiki_dob, pool):
    # Step 1: DOB exact match
    if wiki_dob:
        dob_matches = pool[pool['dob'] == wiki_dob]
        if len(dob_matches) == 1:
            return dob_matches.iloc[0], 'dob_exact'
        elif len(dob_matches) > 1:
            # Multiple same DOB — pick best name match
            scores = dob_matches['name'].apply(
                lambda n: fuzz.token_sort_ratio(wiki_name.lower(), str(n).lower())
            )
            best_idx = scores.idxmax()
            if scores[best_idx] >= 60:
                return dob_matches.loc[best_idx], f'dob+name({scores[best_idx]}%)'

    # Step 2: Name match within nationality pool
    if len(pool) > 0:
        scores = pool['name'].apply(
            lambda n: fuzz.token_sort_ratio(wiki_name.lower(), str(n).lower())
        )
        best_idx   = scores.idxmax()
        best_score = scores[best_idx]
        if best_score >= 75:
            return pool.loc[best_idx], f'name({best_score}%)'

    return None, 'unmatched'

# ─── PROCESS EACH COUNTRY ────────────────────────────────────────────────────
summary_rows = []

for squad in squads:
    if squad['total'] == 0:
        continue

    wiki_country = squad['country']
    country_aliases = COUNTRY_ALIASES.get(wiki_country, [wiki_country])
    normalized_aliases = {normalize_text(alias) for alias in country_aliases}

    # Filter players.csv to this nationality only
    pool = players_df[
        players_df['citizenship_norm'].isin(normalized_aliases)
    ].copy()

    print(f'\n{wiki_country} ({", ".join(country_aliases)}) — {len(pool)} TM players in pool')

    rows = []
    for p in squad['players']:
        wiki_name = p['name']
        wiki_dob  = extract_dob(p.get('dateOfBirth', ''))
        match, method = find_player(wiki_name, wiki_dob, pool)

        if match is not None:
            rows.append({
                'name':          wiki_name,
                'position':      p['position'],
                'club':          p['club'],
                'dob':           wiki_dob,
                'caps':          p['caps'],
                'goals':         p['goals'],
                'tm_name':       match['name'],
                'tm_club':       match['current_club_name'],
                'value_eur':     match['market_value_in_eur'],
                'value_m':       round(float(match['market_value_in_eur'] or 0) / 1_000_000, 2),
                'match_method':  method,
            })
            print(f'  ✅ {wiki_name} → {match["name"]} | €{match["market_value_in_eur"]:,.0f} [{method}]')
        else:
            rows.append({
                'name':          wiki_name,
                'position':      p['position'],
                'club':          p['club'],
                'dob':           wiki_dob,
                'caps':          p['caps'],
                'goals':         p['goals'],
                'tm_name':       None,
                'tm_club':       None,
                'value_eur':     0,
                'value_m':       0,
                'match_method':  'unmatched',
            })
            print(f'  ❌ {wiki_name} — not found')

    # Build country dataframe
    country_df = pd.DataFrame(rows)
    country_df = country_df.sort_values('value_eur', ascending=False)

    # Save individual country file
    safe_name = wiki_country.replace(' ', '_').replace('/', '_')
    write_csv(country_df, OUTPUT_DIR / 'countries' / f'{safe_name}.csv')

    # Summary stats
    total_value    = country_df['value_eur'].sum()
    avg_value      = country_df['value_eur'].mean()
    matched        = country_df['tm_name'].notna().sum()
    top            = country_df.iloc[0]

    print(f'  → Total: €{total_value/1e6:.1f}M | Matched: {matched}/{len(rows)}')

    summary_rows.append({
        'country':            wiki_country,
        'group':              squad['group'],
        'squad_size':         len(rows),
        'matched':            int(matched),
        'match_rate_pct':     round(matched / len(rows) * 100),
        'total_value_m':      round(total_value / 1e6, 1),
        'avg_value_m':        round(avg_value   / 1e6, 2),
        'top_player':         top['name'],
        'top_player_value_m': top['value_m'],
    })

# ─── SUMMARY TABLE ────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows).sort_values('total_value_m', ascending=False)
summary_df.insert(0, 'rank', range(1, len(summary_df) + 1))
summary_path = write_csv(summary_df, OUTPUT_DIR / 'squad_valuations_summary.csv')

top_squads_df = summary_df.head(TOP_BOTTOM_COUNT).copy()
bottom_squads_df = summary_df.sort_values(
    ['total_value_m', 'rank'],
    ascending=[True, True]
).head(TOP_BOTTOM_COUNT).copy()

top_path = write_csv(top_squads_df, OUTPUT_DIR / f'top_{TOP_BOTTOM_COUNT}_squad_valuations.csv')
bottom_path = write_csv(bottom_squads_df, OUTPUT_DIR / f'bottom_{TOP_BOTTOM_COUNT}_squad_valuations.csv')


def print_squad_slice(title, df):
    print(f'\n{title}')
    print('-' * len(title))
    print(f'{"Rank":<5} {"Country":<25} {"Group":<10} {"€M Total":>9} {"Top Player"}')
    for _, r in df.iterrows():
        print(
            f'{int(r["rank"]):<5} {r["country"]:<25} {r["group"]:<10} '
            f'€{r["total_value_m"]:>7.1f}M '
            f'{r["top_player"]} (€{r["top_player_value_m"]}M)'
        )

print('\n' + '='*85)
print(f'{"Rank":<5} {"Country":<25} {"Group":<10} {"€M Total":>9} {"Match%":>7}  {"Top Player"}')
print('='*85)
for _, r in summary_df.iterrows():
    print(
        f'{int(r["rank"]):<5} {r["country"]:<25} {r["group"]:<10} '
        f'€{r["total_value_m"]:>7.1f}M '
        f'{r["match_rate_pct"]:>6}%  '
        f'{r["top_player"]} (€{r["top_player_value_m"]}M)'
    )

print_squad_slice(f'Top {TOP_BOTTOM_COUNT} most valuable squads', top_squads_df)
print_squad_slice(f'Bottom {TOP_BOTTOM_COUNT} least valuable squads', bottom_squads_df.sort_values('total_value_m'))

print(f'\n✅ Done.')
print(f'   Players source   → {players_csv_path}')
print(f'   Individual CSVs → output/countries/')
print(f'   Summary         → output/{summary_path.name}')
print(f'   Top {TOP_BOTTOM_COUNT} CSV       → output/{top_path.name}')
print(f'   Bottom {TOP_BOTTOM_COUNT} CSV    → output/{bottom_path.name}')