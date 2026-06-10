# FIFA WC 2026 - Data Analytics and Machine Learning Projects

Analytics tools for FIFA World Cup 2026, covering player market valuation and referee officiating behavior.

## Projects

### WC Squad Valuations (`src/wc-squad-valuations`)

Builds market-value rankings for all 48 World Cup 2026 squads by matching the announced squad lists from Wikipedia against Transfermarkt player records. The workflow can refresh the latest `players.csv` snapshot from the Kaggle [`davidcariboo/player-scores`](https://www.kaggle.com/datasets/davidcariboo/player-scores) dataset, then computes per-country squad totals, match rates, top players, and the top 3 / bottom 3 most valuable squads.

**Scripts:**
- `wiki-extract.py`: scrapes the latest announced WC 2026 squad lists into `output/wc2026_squads.json`
- `fetch-latest-players.py`: downloads the latest Kaggle `players.csv` into `data/Transfermarkt/kaggle-latest/`
- `build-valuations.py`: matches squad players to Transfermarkt records and writes the valuation outputs

**Outputs:** per-country CSVs in `output/countries/`, `squad_valuations_summary.csv`, `top_3_squad_valuations.csv`, `bottom_3_squad_valuations.csv`.

```bash
python src/wc-squad-valuations/wiki-extract.py
python src/wc-squad-valuations/fetch-latest-players.py --force
python src/wc-squad-valuations/build-valuations.py --refresh-players
```

If `data/Transfermarkt/kaggle-latest/players.csv` already exists, `build-valuations.py` automatically prefers it over the older baseline `data/Transfermarkt/players.csv`. You can also point the build at a specific file with `--players-csv`.

---

### Bubble Detector (`src/bubble-detector`)

Identifies players whose market value may be out of line with recent on-pitch form. The pipeline selects the top 100 players by Transfermarkt market value from World Cup 2026 qualified nations, aggregates their stats over a six-month window, and estimates a form-based valuation using position-specific linear regression (goalkeepers, defenders, midfielders, and forwards use different feature sets). It then compares estimated value to actual market value to surface the largest gaps—potential overvaluations ("bubbles") and undervaluations.

**Outputs:** CSV rankings, paired bar charts (`top_10_valuation_gaps.png`, `top_5_over_and_under_valuation_gaps.png`).

```bash
python src/bubble-detector/bubble-detector.py
```

Optional flags: `--data-dir`, `--output-dir`.

---

### Referee Analysis (`src/referee-analysis`)

Clusters World Cup 2026 referees by behavioral patterns derived from cards and fouls. For each referee, per-competition stats are aggregated into weighted averages (cards per game, fouls per game, card consistency, cards per 100 fouls). K-Means clustering (`k=3`) groups referees with similar officiating styles. The script also evaluates optimal cluster counts (elbow and silhouette), visualizes clusters in PCA space, and optionally generates witty cluster descriptions via a local Ollama model.

**Outputs:** `referee_clusters.json`, `cluster_profiles.json`, `referee_clusters_pca.png`, `elbow_silhouette.png`.

```bash
python src/referee-analysis/cluster_referees.py
```

Requires Ollama running locally for cluster descriptions (optional; errors are captured in output if unavailable).

---

## Datasets

### Transfermarkt (Bubble Detector)

**Location:** `data/Transfermarkt/`

Public football data sourced from [Transfermarkt](https://www.transfermarkt.com/), stored as CSV files. The bubble detector uses:

| File | Key fields |
|------|------------|
| `players.csv` | Player ID, name, citizenship, position, club, market value (EUR) |
| `appearances.csv` | Player ID, match date, goals, assists, minutes played, yellow/red cards |

The analysis filters players to the 48 nations qualified for World Cup 2026, takes the top 100 by market value, and limits appearances to Nov 2025–May 2026 (180-day window ending May 16, 2026).

For `wc-squad-valuations`, the project can also download a newer `players.csv` snapshot from the Kaggle [`davidcariboo/player-scores`](https://www.kaggle.com/datasets/davidcariboo/player-scores) dataset. That file is stored at `data/Transfermarkt/kaggle-latest/players.csv` and is preferred automatically by `build-valuations.py` when present.

---

### Referee Stats (Referee Analysis)

**Location:** `data/referees/`

One JSON file per World Cup 2026 referee (~52 officials). Each file contains a list of season/competition records with officiating statistics:

| Field | Description |
|-------|-------------|
| `season` | Season label (e.g. `2025/2026`) |
| `competition` | League or tournament name |
| `fixtures` | Number of matches officiated |
| `cards` | Average and total cards per game |
| `fouls` | Average and total fouls per game |

Stats span domestic leagues, cups, continental competitions, and international tournaments, giving a cross-competition profile of each referee's card and foul tendencies.
