"""
Create Voronoi diagram example for README
Demonstrates spatial control measurement using Voronoi tessellation
"""

import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import (
    load_match_data,
    get_tracking_dataframe,
    get_possession_info,
    get_ball_dataframe,
)
from src.utils import calculate_velocity
from src.visualization import draw_pitch, plot_players, plot_voronoi
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Configuration
MATCH_ID = "2017461"
FRAME = 13921
PERIOD = 1

# Load data
data = load_match_data(MATCH_ID)
df_vel = calculate_velocity(get_tracking_dataframe(data["tracking"], period=PERIOD))
poss_df = get_possession_info(data["tracking"])
ball_df = get_ball_dataframe(data["tracking"], period=PERIOD)

# Get team IDs
home_ids = list(
    poss_df[poss_df["group"] == "home team"]["player_id"].dropna().astype(int).unique()
)
away_ids = list(
    poss_df[poss_df["group"] == "away team"]["player_id"].dropna().astype(int).unique()
)

# Get frame metadata
timestamp = df_vel[df_vel["frame"] == FRAME].iloc[0]["timestamp"]
minutes, seconds = int(timestamp.split(":")[1]), int(float(timestamp.split(":")[2]))

# Create visualization
fig, ax = plt.subplots(figsize=(14, 10))
draw_pitch(ax)
plot_voronoi(ax, df_vel, FRAME, alpha=0.2)
plot_players(ax, df_vel, FRAME, home_ids, away_ids)

# Ball
ball_pos = ball_df[ball_df["frame"] == FRAME].iloc[0]
ax.scatter(
    ball_pos["x"],
    ball_pos["y"],
    c="white",
    s=150,
    edgecolors="black",
    linewidths=2.5,
    zorder=25,
)

# Ball carrier
bc_id = poss_df[poss_df["frame"] == FRAME].iloc[0]["player_id"]
bc_pos = df_vel[(df_vel["frame"] == FRAME) & (df_vel["player_id"] == bc_id)].iloc[0]
ax.scatter(
    bc_pos["x"],
    bc_pos["y"],
    c="gold",
    s=300,
    marker="s",
    edgecolors="black",
    linewidths=3,
    zorder=20,
)

# Title
title = f"Voronoi Tessellation: Controlled Space per Player\n"
title += (
    f"{data['metadata']['home_team']['name']} {data['metadata']['home_team_score']} - "
)
title += (
    f"{data['metadata']['away_team_score']} {data['metadata']['away_team']['name']}\n"
)
title += f"Period {PERIOD} | {minutes}:{seconds:02d}"
ax.text(
    0.5,
    1.02,
    title,
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=12,
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
            markerfacecolor="blue",
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=1.5,
            label="Home Team (Possession)",
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
            label="Away Team (Defending)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=2,
            label="Ball",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gold",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=2,
            label="Ball Carrier",
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
plt.savefig(figs_dir / "voronoi_example.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"✓ Created: {figs_dir / 'voronoi_example.png'}")
