"""
Verify all statistics mentioned in README and identify top performers
"""

import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_matches_info, load_match_data
from src.space_analysis import analyze_all_matches_normalized
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================
MATCH_IDS = [
    "2017461",  # Melbourne Victory 0-1 Auckland FC
    "1996435",  # Sydney FC 4-1 Adelaide United
    "1886347",  # Wellington Phoenix 2-3 Melbourne Victory
    "1899585",  # Western United 2-2 Sydney FC
    "1925299",  # Central Coast Mariners 3-1 Brisbane Roar
    "1953632",  # Perth Glory 1-2 Adelaide United
    "2006229",  # Newcastle Jets 1-3 Western Sydney Wanderers
    "2011166",  # Melbourne City 2-0 Macarthur FC
    "2013725",  # Western United 1-1 Perth Glory
    "2015213",  # Brisbane Roar 2-1 Newcastle Jets
]

# How many matches to analyze (1-10)
NUM_MATCHES = 10  # ← CHANGE THIS

MATCH_IDS = MATCH_IDS[:NUM_MATCHES]
# ============================================================================

print(f"Analyzing {len(MATCH_IDS)} matches...")
print(f"Match IDs: {MATCH_IDS}\n")

matches = load_matches_info(MATCH_IDS)
trajectories = analyze_all_matches_normalized(
    matches, load_match_data, velocity_threshold=5.0
)

print("\n" + "=" * 80)
print("RESULTS VERIFICATION")
print("=" * 80)

# Total runs
total_runs = len(trajectories)
avg_runs_per_match = total_runs / len(MATCH_IDS)
print(f"\nTotal runs: {total_runs}")
print(f"Average runs per match: {avg_runs_per_match:.1f}")

# Space created
total_space = trajectories["total_space_created"].sum()
avg_space_per_run = trajectories["total_space_created"].mean()
pitch_area = 105 * 68  # 7,140 m²
pct_of_pitch = (avg_space_per_run / pitch_area) * 100
print(f"\nTotal space created: {total_space:,.0f} m²")
print(f"Average space per run: {avg_space_per_run:,.0f} m²")
print(f"Percentage of pitch: {pct_of_pitch:.1f}%")

# Velocity
avg_velocity = trajectories["max_velocity"].mean()
avg_velocity_kmh = avg_velocity * 3.6
print(f"\nAverage velocity: {avg_velocity:.2f} m/s ({avg_velocity_kmh:.1f} km/h)")

# Duration
avg_duration_frames = trajectories["duration_frames"].mean()
avg_duration_seconds = avg_duration_frames / 10
print(
    f"\nAverage duration: {avg_duration_seconds:.1f} seconds ({avg_duration_frames:.1f} frames)"
)

# Build player name dictionary
print("\nLoading player names...")
all_player_names = {}

for match_id in trajectories["match_id"].unique():
    data = load_match_data(match_id)

    for p in data["metadata"]["players"]:
        player_id = p["id"]
        first = p.get("first_name", "")
        last = p.get("last_name", "")
        team_id = p["team_id"]
        home_id = data["metadata"]["home_team"]["id"]

        team_name = (
            data["metadata"]["home_team"]["name"]
            if team_id == home_id
            else data["metadata"]["away_team"]["name"]
        )

        all_player_names[player_id] = {
            "name": f"{first} {last}".strip(),
            "team": team_name,
        }

print(f"Loaded {len(all_player_names)} player names")

# Player statistics
print("\nCalculating player statistics...")

all_player_stats = (
    trajectories.groupby("player_id")
    .agg({"total_space_created": ["count", "sum", "mean"], "match_id": "nunique"})
    .reset_index()
)

all_player_stats.columns = [
    "player_id",
    "num_runs",
    "total_space",
    "avg_space_per_run",
    "matches_played",
]

all_player_stats["runs_per_match"] = (
    all_player_stats["num_runs"] / all_player_stats["matches_played"]
)
all_player_stats["space_per_match"] = (
    all_player_stats["total_space"] / all_player_stats["matches_played"]
)

print(f"Total unique players: {len(all_player_stats)}")

print("\n" + "=" * 80)
print("TOP PERFORMERS")
print("=" * 80)

# 1. Most runs per match
top_runs = all_player_stats.nlargest(1, "runs_per_match").iloc[0]
pid = int(top_runs["player_id"])
pinfo = all_player_names.get(pid, {"name": f"Player {pid}", "team": "Unknown"})
print(f"\n1. MOST RUNS/MATCH: {pinfo['name']} ({pinfo['team']})")
print(
    f"   {top_runs['runs_per_match']:.1f} runs/match ({int(top_runs['num_runs'])} runs in {int(top_runs['matches_played'])} matches) | {top_runs['space_per_match']:,.0f} m²/match | {top_runs['avg_space_per_run']:,.0f} m²/run"
)

# 2. Most space per match
top_space = all_player_stats.nlargest(1, "space_per_match").iloc[0]
pid = int(top_space["player_id"])
pinfo = all_player_names.get(pid, {"name": f"Player {pid}", "team": "Unknown"})
print(f"\n2. MOST SPACE/MATCH: {pinfo['name']} ({pinfo['team']})")
print(
    f"   {top_space['space_per_match']:,.0f} m²/match ({top_space['total_space']:,.0f} m² in {int(top_space['matches_played'])} matches) | {top_space['runs_per_match']:.1f} runs/match | {top_space['avg_space_per_run']:,.0f} m²/run"
)

# 3. Most efficient (min 5 runs total)
all_player_stats_eff = all_player_stats[all_player_stats["num_runs"] >= 5]
if len(all_player_stats_eff) > 0:
    top_eff = all_player_stats_eff.nlargest(1, "avg_space_per_run").iloc[0]
    pid = int(top_eff["player_id"])
    pinfo = all_player_names.get(pid, {"name": f"Player {pid}", "team": "Unknown"})
    print(f"\n3. MOST EFFICIENT (min 5 runs): {pinfo['name']} ({pinfo['team']})")
    print(
        f"   {top_eff['avg_space_per_run']:,.0f} m²/run ({int(top_eff['num_runs'])} runs in {int(top_eff['matches_played'])} matches) | {top_eff['runs_per_match']:.1f} runs/match | {top_eff['space_per_match']:,.0f} m²/match"
    )
else:
    print(f"\n3. MOST EFFICIENT: Not enough data (need min 5 runs)")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
