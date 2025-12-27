"""
Create trajectory visualization example for README
Demonstrates all off-ball runs from selected team with normalized attack direction
"""

import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_match_data, get_tracking_dataframe, get_possession_info
from src.utils import calculate_velocity
from src.space_analysis import analyze_offball_runs, group_runs_to_trajectories
from src.visualization import draw_pitch, plot_run_trajectories
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Configuration
MATCH_ID = "2011166"
VELOCITY_THRESHOLD = 5.0
TEAM = "away"

# Load data
data = load_match_data(MATCH_ID)
poss_df = get_possession_info(data["tracking"])

# Get team info
home_ids = list(
    poss_df[poss_df["group"] == "home team"]["player_id"].dropna().astype(int).unique()
)
away_ids = list(
    poss_df[poss_df["group"] == "away team"]["player_id"].dropna().astype(int).unique()
)
home_name = data["metadata"]["home_team"]["name"]
away_name = data["metadata"]["away_team"]["name"]
match_info = f"{home_name} {data['metadata']['home_team_score']} - {data['metadata']['away_team_score']} {away_name}"

# Select team
selected_name = home_name if TEAM == "home" else away_name
selected_color = "blue" if TEAM == "home" else "red"

# Process both periods
all_traj = []
for period in [1, 2]:
    df = get_tracking_dataframe(data["tracking"], period=period)
    if len(df) == 0:
        continue

    df_vel = calculate_velocity(df)
    runs = analyze_offball_runs(
        df_vel, poss_df, home_ids, away_ids, velocity_threshold=VELOCITY_THRESHOLD
    )
    runs = runs.merge(
        df_vel[["frame", "player_id", "x", "y"]], on=["frame", "player_id"]
    )
    traj = group_runs_to_trajectories(runs)

    if len(traj) > 0:
        # Normalize attack direction
        home_team_side = data["metadata"].get("home_team_side", [])
        if len(home_team_side) > period - 1:
            if home_team_side[period - 1] == "right_to_left":
                traj.loc[traj["team"] == "home", ["start_x", "end_x"]] *= -1
            else:
                traj.loc[traj["team"] == "away", ["start_x", "end_x"]] *= -1

        all_traj.append(traj)

# Combine and filter selected team
trajectories = pd.concat(all_traj, ignore_index=True)
selected_traj = trajectories[trajectories["team"] == TEAM]

# Create figure
fig, ax = plt.subplots(figsize=(16, 10))
draw_pitch(ax)
plot_run_trajectories(
    ax, selected_traj, color=selected_color, alpha=0.65, velocity_colormap=False
)

# Title
title = f"{match_info}\n{selected_name}: {len(selected_traj)} off-ball runs, {selected_traj['total_space_created'].sum():,.0f} m² space created"
ax.text(
    0.5,
    1.02,
    title,
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=13,
    weight="bold",
)

# Legend
ax.legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=1.5,
            label="Run Start",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=1.5,
            label="Run End",
        ),
    ],
    loc="upper left",
    fontsize=11,
    framealpha=0.9,
)

# Create figs directory in parent (main) directory
figs_dir = Path(__file__).parent.parent / "figs"
figs_dir.mkdir(exist_ok=True)

plt.tight_layout()
plt.savefig(figs_dir / "trajectory_visualization.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"✓ Created: {figs_dir / 'trajectory_visualization.png'}")
print(f"  Team: {selected_name} ({TEAM})")
print(f"  Runs: {len(selected_traj)}")
