"""
Market Bubble Detector - PRD 2.0
Analyzes top 100 international players' 6-month form vs. market valuation.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# PRD 2.0 Target nationalities across 6 confederations
WORLD_CUP_2026_NATIONS = {
    # AFC (9)
    "Australia", "Iran", "Iraq", "Japan", "Jordan", "Qatar",
    "Saudi Arabia", "South Korea", "Uzbekistan",
    # CAF (10)
    "Algeria", "Cape Verde", "DR Congo", "Egypt", "Ghana",
    "Ivory Coast", "Morocco", "Senegal", "South Africa", "Tunisia",
    # CONCACAF (6)
    "Canada", "Curaçao", "Haiti", "Mexico", "Panama", "United States",
    # CONMEBOL (5)
    "Argentina", "Brazil", "Colombia", "Ecuador", "Paraguay", "Uruguay",
    # OFC (1)
    "New Zealand",
    # UEFA (17)
    "Austria", "Belgium", "Bosnia and Herzegovina", "Croatia",
    "Czech Republic", "England", "France", "Germany", "Netherlands",
    "Norway", "Portugal", "Scotland", "Spain", "Sweden", "Switzerland", "Turkey",
}

COLOR_PALETTE = {
    "green": "#3CAC3B",
    "blue": "#2A398D",
    "red": "#E61D25",
    "light_gray": "#D1D4D1",
    "dark_gray": "#474A4A",
}

COUNTRY_ALIASES = {
    "usa": "United States",
    "cote d'ivoire": "Ivory Coast",
    "congo dr": "DR Congo",
    "dr congo": "DR Congo",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "curacao": "Curaçao",
    "south korea": "South Korea",
    "czechia": "Czech Republic",
}


@dataclass
class PlayerFormMetrics:
    player_id: int
    name: str
    country: str
    position: str
    actual_value_eur: float
    games_6m: int
    minutes_6m: float
    goals_6m: float
    assists_6m: float
    shots_6m: float
    key_passes_6m: float
    cards_6m: float
    goals_per90_6m: float
    assists_per90_6m: float


def normalize_country(value: object) -> str:
    text = str(value).strip().lower() if value else ""
    return COUNTRY_ALIASES.get(text, str(value).strip() if value else "")


def resolve_data_dir(start: Optional[Path] = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    candidates = [current.parent, current.parent.parent, current.parent.parent.parent]
    for candidate in candidates:
        transfermarkt = candidate / "data" / "Transfermarkt"
        if transfermarkt.exists():
            return transfermarkt
    raise FileNotFoundError("Could not locate data/Transfermarkt")


def infer_position_group(position: object, sub_position: object) -> str:
    raw = f"{position or ''} {sub_position or ''}".strip().lower()
    if "goalkeeper" in raw or "keeper" in raw:
        return "Goalkeeper"
    if any(x in raw for x in ["defender", "centre-back", "left-back", "right-back", "wing-back"]):
        return "Defender"
    if any(x in raw for x in ["midfielder", "midfield", "cm", "cam", "cdm"]):
        return "Midfielder"
    if any(x in raw for x in ["forward", "striker", "winger", "attacking"]):
        return "Forward"
    return "Midfielder"


def load_and_select_top_100(data_dir: Path) -> pd.DataFrame:
    """Load players and select top 100 by market value from target nations."""
    players = pd.read_csv(
        data_dir / "players.csv",
        usecols=[
            "player_id", "name", "country_of_citizenship", "position",
            "sub_position", "market_value_in_eur", "current_club_name",
        ]
    )

    # Normalize country names
    players["country_of_citizenship"] = players["country_of_citizenship"].map(
        lambda x: COUNTRY_ALIASES.get(
            str(x).strip().lower() if x else "",
            str(x).strip() if x else ""
        )
    )

    # Filter to target nations
    players = players[players["country_of_citizenship"].isin(WORLD_CUP_2026_NATIONS)].copy()

    # Convert market value to numeric
    players["market_value_in_eur"] = pd.to_numeric(
        players["market_value_in_eur"], errors="coerce"
    ).fillna(0)

    # Sort by market value descending and select top 100
    players = players.sort_values("market_value_in_eur", ascending=False).head(100)

    # Infer position group
    players["position_group"] = players.apply(
        lambda row: infer_position_group(row.get("position"), row.get("sub_position")), axis=1
    )

    return players.reset_index(drop=True)


def extract_6month_appearances(data_dir: Path, player_ids: list[int]) -> pd.DataFrame:
    """Extract appearances strictly for past 6 months (Nov 2025 - May 2026)."""
    appearances = pd.read_csv(
        data_dir / "appearances.csv",
        usecols=[
            "player_id", "date", "goals", "assists", "minutes_played",
            "yellow_cards", "red_cards",
        ]
    )

    # Parse dates
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")

    # Reference date is May 16, 2026
    reference_date = pd.Timestamp("2026-05-16")
    six_months_back = reference_date - timedelta(days=180)

    # Filter to date range
    appearances = appearances[
        (appearances["date"] >= six_months_back) &
        (appearances["date"] <= reference_date)
    ].copy()

    # Filter to player pool
    appearances = appearances[appearances["player_id"].isin(player_ids)].copy()

    return appearances


def calculate_form_metrics(
    players_df: pd.DataFrame,
    appearances_6m: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate 6-month form metrics by player."""
    form_agg = (
        appearances_6m.groupby("player_id", as_index=False)
        .agg(
            games_6m=("player_id", "count"),
            minutes_6m=("minutes_played", "sum"),
            goals_6m=("goals", "sum"),
            assists_6m=("assists", "sum"),
            cards_6m=("yellow_cards", lambda x: x.sum() + (appearances_6m.loc[x.index, "red_cards"].sum() * 2)),
        )
    )

    # Merge with player data
    players_with_form = players_df.merge(form_agg, left_on="player_id", right_on="player_id", how="left")

    # Fill missing games with 0 (no appearances in 6 months)
    players_with_form["games_6m"] = players_with_form["games_6m"].fillna(0)
    players_with_form["minutes_6m"] = players_with_form["minutes_6m"].fillna(0)
    players_with_form["goals_6m"] = players_with_form["goals_6m"].fillna(0)
    players_with_form["assists_6m"] = players_with_form["assists_6m"].fillna(0)
    players_with_form["cards_6m"] = players_with_form["cards_6m"].fillna(0)

    # Calculate per-90 metrics
    players_with_form["goals_per90_6m"] = np.where(
        players_with_form["minutes_6m"] > 0,
        (players_with_form["goals_6m"] / players_with_form["minutes_6m"]) * 90,
        0
    )
    players_with_form["assists_per90_6m"] = np.where(
        players_with_form["minutes_6m"] > 0,
        (players_with_form["assists_6m"] / players_with_form["minutes_6m"]) * 90,
        0
    )

    return players_with_form


def estimate_valuation_by_position(players_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate player valuations using position-specific regression models."""
    players_df = players_df.copy()
    players_df["estimated_value_eur"] = 0.0

    for position in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
        pos_players = players_df[players_df["position_group"] == position].copy()

        if len(pos_players) < 3:
            continue

        # Features based on position
        if position == "Goalkeeper":
            feature_cols = ["minutes_6m", "cards_6m"]
        elif position == "Defender":
            feature_cols = ["minutes_6m", "cards_6m", "assists_per90_6m"]
        elif position == "Midfielder":
            feature_cols = ["minutes_6m", "goals_per90_6m", "assists_per90_6m", "cards_6m"]
        else:  # Forward
            feature_cols = ["minutes_6m", "goals_per90_6m", "assists_per90_6m"]

        # Filter to available features
        available_features = [col for col in feature_cols if col in pos_players.columns]
        if not available_features:
            continue

        X = pos_players[available_features].fillna(0).values
        y = pos_players["market_value_in_eur"].values

        # Only train if we have sufficient samples and variance
        if len(y) >= 3 and np.std(y) > 0:
            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                model = LinearRegression()
                model.fit(X_scaled, y)

                # Predict for this position group
                pred = model.predict(X_scaled)
                pred = np.maximum(pred, 1)  # Floor at €1

                players_df.loc[pos_players.index, "estimated_value_eur"] = pred
            except Exception:
                # Fallback: use mean value
                players_df.loc[pos_players.index, "estimated_value_eur"] = np.mean(y)
        else:
            # Fallback: use median value per position
            players_df.loc[pos_players.index, "estimated_value_eur"] = np.median(y) if len(y) > 0 else 1

    return players_df


def calculate_valuation_gap(players_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate valuation difference and identify top 10."""
    players_df = players_df.copy()

    players_df["valuation_difference"] = (
        players_df["market_value_in_eur"] - players_df["estimated_value_eur"]
    )
    players_df["abs_valuation_difference"] = np.abs(players_df["valuation_difference"])

    # Sort by absolute difference and get top 10
    top_10 = players_df.nlargest(10, "abs_valuation_difference")

    return top_10.reset_index(drop=True)


def calculate_top5_over_under(players_df: pd.DataFrame) -> pd.DataFrame:
    """Return top 5 overvalued (positive diff) and top 5 undervalued (negative diff).

    This is the signed selection you asked for: 5 largest positive valuation differences
    and 5 largest negative valuation differences (most undervalued).
    """
    df = players_df.copy()
    df["valuation_difference"] = df["market_value_in_eur"] - df["estimated_value_eur"]

    over = df[df["valuation_difference"] > 0].nlargest(5, "valuation_difference")
    under = df[df["valuation_difference"] < 0].nsmallest(5, "valuation_difference")

    combined = pd.concat([over, under])
    combined["abs_valuation_difference"] = combined["valuation_difference"].abs()
    return combined.reset_index(drop=True)


def create_paired_bar_chart(top_10: pd.DataFrame, output_path: Path) -> None:
    """Create paired bar chart for top 10 players with custom color palette."""
    if top_10.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Sort by actual market value ascending for consistent visual ordering
    top_10 = top_10.sort_values("market_value_in_eur", ascending=True)

    x = np.arange(len(top_10))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))

    # Determine colors based on over/undervaluation
    colors_actual = []
    colors_estimated = []
    for _, row in top_10.iterrows():
        if row["valuation_difference"] > 0:
            # Overvalued (bubble)
            colors_actual.append(COLOR_PALETTE["red"])
            colors_estimated.append(COLOR_PALETTE["light_gray"])
        else:
            # Undervalued
            colors_actual.append(COLOR_PALETTE["green"])
            colors_estimated.append(COLOR_PALETTE["light_gray"])

    # Actual value bars
    bars1 = ax.bar(
        x - width / 2,
        top_10["market_value_in_eur"] / 1_000_000,
        width,
        label="Actual Market Value",
        color=colors_actual,
        edgecolor=COLOR_PALETTE["dark_gray"],
        linewidth=0.5,
    )

    # Estimated value bars
    bars2 = ax.bar(
        x + width / 2,
        top_10["estimated_value_eur"] / 1_000_000,
        width,
        label="Estimated Value (6M Form)",
        color=colors_estimated,
        edgecolor=COLOR_PALETTE["dark_gray"],
        linewidth=0.5,
    )

    # Labels and formatting
    ax.set_xlabel("Player", fontsize=11, color=COLOR_PALETTE["dark_gray"], fontweight="bold")
    ax.set_ylabel("Value (EUR, millions)", fontsize=11, color=COLOR_PALETTE["dark_gray"], fontweight="bold")
    ax.set_title(
        "Top 10 Valuation Gaps: 6-Month Form vs. Market Value",
        fontsize=13,
        color=COLOR_PALETTE["dark_gray"],
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{name}\n{country}" for name, country in zip(top_10["name"], top_10["country_of_citizenship"])],
        fontsize=9,
        color=COLOR_PALETTE["dark_gray"],
    )

    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_axisbelow(True)

    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"€{height:.1f}M",
                ha="center",
                va="bottom",
                fontsize=8,
                color=COLOR_PALETTE["dark_gray"],
            )

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"€{height:.1f}M",
                ha="center",
                va="bottom",
                fontsize=8,
                color=COLOR_PALETTE["dark_gray"],
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Saved visualization to {output_path}")


def export_results(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "top_10_valuation_gaps.csv",
    title: str = "Top 10 Valuation Gaps",
    sort_by: str = "market_value_in_eur",
    ascending: bool = True,
    print_table: bool = True,
) -> None:
    """Export results to CSV and print a titled table.

    Parameters:
    - df: DataFrame to export
    - output_dir: target directory
    - filename: CSV filename
    - title: printed table title
    - sort_by: column to sort the exported/printed table by
    - ascending: sort order
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / filename
    export_cols = [
        "name", "country_of_citizenship", "position_group",
        "market_value_in_eur", "estimated_value_eur",
        "valuation_difference", "abs_valuation_difference",
        "games_6m", "minutes_6m", "goals_6m", "assists_6m",
        "goals_per90_6m", "assists_per90_6m",
    ]

    top_export = df[export_cols].copy()

    # Sort exported results by requested column
    if sort_by in top_export.columns:
        top_export = top_export.sort_values(sort_by, ascending=ascending)

    top_export.to_csv(output_csv, index=False)

    print(f"\n✅ Results exported to {output_csv}")
    if print_table:
        print(f"\n{title}:")
        print(top_export.to_string(index=False))


def run_pipeline(data_dir: Path, output_dir: Path) -> None:
    """Execute the full bubble detector pipeline."""
    print("🔍 Loading top 100 players from target nations...")
    players_df = load_and_select_top_100(data_dir)
    print(f"   Loaded {len(players_df)} players")

    print("\n📅 Extracting 6-month appearances (Nov 2025 - May 2026)...")
    appearances_6m = extract_6month_appearances(data_dir, players_df["player_id"].tolist())
    print(f"   Found {len(appearances_6m)} appearance records")

    print("\n📊 Calculating 6-month form metrics...")
    players_with_form = calculate_form_metrics(players_df, appearances_6m)

    print("\n🎯 Estimating valuations based on 6-month form...")
    players_with_est = estimate_valuation_by_position(players_with_form)

    print("\n💰 Calculating valuation gaps...")
    top_10 = calculate_valuation_gap(players_with_est)

    print("\n🎨 Creating paired bar chart visualization...")
    viz_path = output_dir / "top_10_valuation_gaps.png"
    create_paired_bar_chart(top_10, viz_path)
    print(f"   Saved visualization to {viz_path}")

    # Alternative selection: top 5 overvalued and top 5 undervalued (signed selection)
    signed_top10 = calculate_top5_over_under(players_with_est)
    viz_signed = output_dir / "top_5_over_and_under_valuation_gaps.png"
    create_paired_bar_chart(signed_top10, viz_signed)
    print(f"   Saved alternative visualization to {viz_signed}")
    export_results(
        signed_top10,
        output_dir / "signed_top5",
        filename="signed_top5_valuation_gaps.csv",
        title="Top 5 Overvalued and Top 5 Undervalued",
        sort_by="market_value_in_eur",
        ascending=True,
    )

    print("\n💾 Exporting results...")
    # Export absolute top-10 but do not print table to avoid confusion with the signed visualization
    export_results(
        top_10,
        output_dir,
        filename="top_10_valuation_gaps.csv",
        title="Top 10 Valuation Gaps (Absolute)",
        sort_by="market_value_in_eur",
        ascending=True,
        print_table=False,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Market Bubble Detector - PRD 2.0"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Path to Transfermarkt data directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for results"
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data_dir = args.data_dir or resolve_data_dir()
    output_dir = args.output_dir or (Path(__file__).resolve().parent / "output")

    print("=" * 70)
    print("Market Bubble Detector - PRD 2.0")
    print("Analyzing Top 100 Players Over 6-Month Form Window")
    print("=" * 70)

    run_pipeline(data_dir, output_dir)

    print("\n" + "=" * 70)
    print("✨ Pipeline Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
