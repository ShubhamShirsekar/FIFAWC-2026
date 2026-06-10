import requests
import json
import re
import os
from bs4 import BeautifulSoup

URL     = 'https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
OUTPUT  = './output/wc2026_squads.json'

os.makedirs('./output', exist_ok=True)

# ─── FETCH PAGE ───────────────────────────────────────────────────────────────
print('Fetching Wikipedia squads page...')
response = requests.get(URL, headers=HEADERS)
soup     = BeautifulSoup(response.content, 'lxml')
print(f'Status: {response.status_code}')

# ─── PARSE SQUADS ─────────────────────────────────────────────────────────────
def parse_age(dob_text):
    match = re.search(r'age\s+(\d+)', dob_text)
    return int(match.group(1)) if match else None

def parse_dob(dob_text):
    # Extract date before the parenthesis
    match = re.search(r'(\d{1,2}\s+\w+\s+\d{4})', dob_text)
    return match.group(1).strip() if match else dob_text.strip()

def parse_player_table(table):
    players = []
    rows    = table.find_all('tr')[1:]  # skip header

    for row in rows:
        cols = row.find_all(['td', 'th'])
        if len(cols) < 7:
            continue

        dob_raw = cols[3].get_text().strip()

        players.append({
            'number':      cols[0].get_text().strip(),
            'position':    cols[1].get_text().strip(),
            'name':        cols[2].get_text().strip(),
            'dateOfBirth': parse_dob(dob_raw),
            'age':         parse_age(dob_raw),
            'caps':        cols[4].get_text().strip(),
            'goals':       cols[5].get_text().strip(),
            'club':        cols[6].get_text().strip(),
        })

    return players

# ─── WALK THROUGH HEADINGS ────────────────────────────────────────────────────
all_squads   = []
current_group = None

# Wikipedia now nests squad content inside #mw-content-text, not the
# top-level .mw-parser-output div (which only holds the protection icon).
content = soup.find('div', id='mw-content-text')
if not content:
    raise RuntimeError('Could not find Wikipedia article content')

elements = content.find_all(['h2', 'h3', 'table'])

i = 0
while i < len(elements):
    el = elements[i]

    # Track current group from h2
    if el.name == 'h2':
        text = el.get_text().strip().replace('[edit]', '').strip()
        if text.startswith('Group'):
            current_group = text
        elif text.startswith('Statistics'):
            current_group = None

    # Country found in h3
    elif el.name == 'h3':
        country = el.get_text().replace('[edit]', '').strip()

        # Skip non-country headings and anything after Statistics
        skip = ['Contents', 'References', 'External', 'Statistics']
        if current_group is None or any(s in country for s in skip):
            i += 1
            continue

        # Look for the next wikitable after this h3
        table = None
        for j in range(i + 1, min(i + 5, len(elements))):
            candidate = elements[j]
            if candidate.name == 'table' and 'wikitable' in candidate.get('class', []):
                table = candidate
                break

        if table:
            players = parse_player_table(table)
            squad = {
                'group':   current_group,
                'country': country,
                'players': players,
                'total':   len(players)
            }
            all_squads.append(squad)
            print(f'✅ {current_group} | {country}: {len(players)} players')
        else:
            print(f'⏳ {current_group} | {country}: squad not announced yet')
            all_squads.append({
                'group':   current_group,
                'country': country,
                'players': [],
                'total':   0
            })

    i += 1

# ─── SAVE OUTPUT ──────────────────────────────────────────────────────────────
with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(all_squads, f, indent=2, ensure_ascii=False)

announced = [s for s in all_squads if s['total'] > 0]
pending   = [s for s in all_squads if s['total'] == 0]

print(f'\n✅ Done.')
print(f'   Squads announced:  {len(announced)}')
print(f'   Squads pending:    {len(pending)}')
print(f'   Total players:     {sum(s["total"] for s in all_squads)}')
print(f'   Saved to:          {OUTPUT}')

if pending:
    print(f'\n⏳ Still pending:')
    for s in pending:
        print(f'   - {s["country"]} ({s["group"]})')