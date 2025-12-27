"""
Visualization functions for football pitch and tracking data
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


def draw_pitch(
    ax=None, pitch_length=105, pitch_width=68, color="white", linecolor="black"
):
    """
    Draw a regulation football pitch.

    Args:
        ax: Matplotlib axis (creates new if None)
        pitch_length (float): Length in meters (default: 105)
        pitch_width (float): Width in meters (default: 68)
        color (str): Pitch background color
        linecolor (str): Line color

    Returns:
        matplotlib.axes.Axes: Configured axis with pitch
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    half_length = pitch_length / 2
    half_width = pitch_width / 2

    # Pitch outline
    ax.add_patch(
        patches.Rectangle(
            (-half_length, -half_width),
            pitch_length,
            pitch_width,
            facecolor=color,
            edgecolor=linecolor,
            linewidth=2,
        )
    )

    # Center line
    ax.plot([0, 0], [-half_width, half_width], color=linecolor, linewidth=2)

    # Center circle
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color=linecolor, linewidth=2))

    # Penalty boxes
    ax.add_patch(
        patches.Rectangle(
            (-half_length, -20.15),
            16.5,
            40.3,
            fill=False,
            edgecolor=linecolor,
            linewidth=2,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (half_length - 16.5, -20.15),
            16.5,
            40.3,
            fill=False,
            edgecolor=linecolor,
            linewidth=2,
        )
    )

    # Goal boxes
    ax.add_patch(
        patches.Rectangle(
            (-half_length, -9.16),
            5.5,
            18.32,
            fill=False,
            edgecolor=linecolor,
            linewidth=2,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (half_length - 5.5, -9.16),
            5.5,
            18.32,
            fill=False,
            edgecolor=linecolor,
            linewidth=2,
        )
    )

    ax.set_xlim(-half_length - 5, half_length + 5)
    ax.set_ylim(-half_width - 5, half_width + 5)
    ax.set_aspect("equal")
    ax.axis("off")

    return ax


def add_match_info(ax, match_data, frame_number, df):
    """
    Add match information to plot.

    Args:
        ax: Matplotlib axis
        match_data (dict): Match metadata
        frame_number (int): Current frame
        df (pd.DataFrame): Tracking data with timestamp
    """
    home_name = match_data["metadata"]["home_team"]["name"]
    away_name = match_data["metadata"]["away_team"]["name"]
    home_score = match_data["metadata"]["home_team_score"]
    away_score = match_data["metadata"]["away_team_score"]

    # Get timestamp
    frame_data = df[df["frame"] == frame_number]
    if len(frame_data) > 0:
        timestamp = frame_data.iloc[0]["timestamp"]
        period = frame_data.iloc[0]["period"]

        # Convert timestamp to minutes:seconds
        time_parts = timestamp.split(":")
        minutes = int(time_parts[1])
        seconds = int(float(time_parts[2]))

        match_info = f"{home_name} {home_score} - {away_score} {away_name}\n"
        match_info += f"Period {period} | {minutes}:{seconds:02d}"
    else:
        match_info = f"{home_name} {home_score} - {away_score} {away_name}"

    ax.text(
        0.5,
        1.02,
        match_info,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold",
    )


def plot_players(
    ax,
    df,
    frame_number,
    home_player_ids=None,
    away_player_ids=None,
    show_ids=False,
    show_ball=False,
    ball_df=None,
):
    """
    Plot player positions for a single frame.

    Args:
        ax: Matplotlib axis
        df (pd.DataFrame): Tracking data
        frame_number (int): Frame to plot
        home_player_ids (list, optional): Home team player IDs (colored blue)
        away_player_ids (list, optional): Away team player IDs (colored red)
        show_ids (bool): Display player IDs as text
        show_ball (bool): Display ball position
        ball_df (pd.DataFrame, optional): Ball tracking data

    Returns:
        matplotlib.axes.Axes: Updated axis
    """
    frame_df = df[df["frame"] == frame_number]

    if len(frame_df) == 0:
        return ax

    # Determine colors by team
    if home_player_ids is not None and away_player_ids is not None:
        colors = []
        for pid in frame_df["player_id"]:
            if pid in home_player_ids:
                colors.append("blue")
            elif pid in away_player_ids:
                colors.append("red")
            else:
                colors.append("gray")
    else:
        colors = "blue"

    # Plot players
    ax.scatter(
        frame_df["x"],
        frame_df["y"],
        c=colors,
        s=200,
        alpha=0.7,
        edgecolors="black",
        linewidths=1.5,
        zorder=10,
    )

    # Optional player IDs
    if show_ids:
        for _, player in frame_df.iterrows():
            ax.text(
                player["x"],
                player["y"],
                str(player["player_id"])[-4:],
                ha="center",
                va="center",
                fontsize=6,
                weight="bold",
                color="white",
            )

    # Plot ball
    if show_ball and ball_df is not None:
        ball_frame = ball_df[ball_df["frame"] == frame_number]
        if len(ball_frame) > 0:
            ax.scatter(
                ball_frame["x"].iloc[0],
                ball_frame["y"].iloc[0],
                c="white",
                s=100,
                edgecolors="black",
                linewidths=2,
                marker="o",
                zorder=15,
                label="Ball",
            )

    return ax


def plot_voronoi(ax, df, frame_number, alpha=0.3):
    """
    Overlay Voronoi diagram on pitch.

    Args:
        ax: Matplotlib axis
        df (pd.DataFrame): Tracking data
        frame_number (int): Frame to plot
        alpha (float): Region transparency

    Returns:
        matplotlib.axes.Axes: Updated axis
    """
    from scipy.spatial import Voronoi

    frame_df = df[df["frame"] == frame_number]

    if len(frame_df) < 4:
        return ax

    points = frame_df[["x", "y"]].values

    try:
        vor = Voronoi(points)
    except:
        return ax

    # Draw regions
    for region in vor.regions:
        if -1 not in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), alpha=alpha, edgecolor="black", linewidth=0.5)

    return ax


def plot_space_creation_heatmap(ax, runs_df, bins=15, team_name=None, match_info=None):
    """
    Create heatmap of space creation locations.

    Args:
        ax: Matplotlib axis
        runs_df (pd.DataFrame): Runs with x, y, space_created columns
        bins (int): Histogram bins (default: 15)
        team_name (str, optional): Team name for title
        match_info (str, optional): Match info string (e.g., "Team A 2-1 Team B")

    Returns:
        tuple: (axis, image) or (axis, None) if error
    """
    if len(runs_df) == 0 or "x" not in runs_df.columns:
        return ax, None

    # Calculate total space created
    total_space = runs_df["space_created"].sum()
    num_runs = len(runs_df)

    # Add match info and stats as title
    if match_info and team_name:
        title = (
            f"{match_info}\n{team_name}: {num_runs} runs, {total_space:,.0f} m² created"
        )
        ax.text(
            0.5,
            1.02,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    # 2D histogram weighted by space created
    heatmap, xedges, yedges = np.histogram2d(
        runs_df["x"], runs_df["y"], bins=bins, weights=runs_df["space_created"]
    )

    # Normalize
    heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap

    # Plot heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(
        heatmap.T,
        extent=extent,
        origin="lower",
        cmap="Reds",
        alpha=0.7,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    # Overlay scatter
    ax.scatter(
        runs_df["x"],
        runs_df["y"],
        c=runs_df["space_created"],
        s=50,
        cmap="Reds",
        alpha=0.5,
        edgecolors="black",
        linewidths=0.5,
    )

    return ax, im


def plot_run_trajectories(
    ax,
    trajectories_df,
    color="blue",
    alpha=0.6,
    top_percentile=None,
    velocity_colormap=False,
):
    """
    Plot run trajectories as arrows from start to end position.

    Args:
        ax: Matplotlib axis
        trajectories_df (pd.DataFrame): Grouped trajectories with start_x, start_y, end_x, end_y
        color (str): Arrow color (if not using velocity colormap)
        alpha (float): Arrow transparency
        top_percentile (float, optional): Only show top X% by space_created (e.g., 0.25 for top 25%)
        velocity_colormap (bool): Color arrows by max_velocity

    Returns:
        matplotlib.axes.Axes: Updated axis
    """
    if len(trajectories_df) == 0:
        return ax

    df = trajectories_df.copy()

    # Filter to top percentile if specified
    if top_percentile is not None:
        threshold = df["total_space_created"].quantile(1 - top_percentile)
        df = df[df["total_space_created"] >= threshold]

    # Calculate arrow components
    df["dx"] = df["end_x"] - df["start_x"]
    df["dy"] = df["end_y"] - df["start_y"]

    # Plot trajectories
    for idx, row in df.iterrows():
        # Determine color
        if velocity_colormap and "max_velocity" in df.columns:
            # Normalize velocity to 0-1 range for colormap
            vel_normalized = (row["max_velocity"] - 5.0) / (
                12.0 - 5.0
            )  # 5-12 m/s range
            vel_normalized = np.clip(vel_normalized, 0, 1)
            arrow_color = plt.cm.Reds(vel_normalized)
        else:
            arrow_color = color

        # Draw arrow
        ax.arrow(
            row["start_x"],
            row["start_y"],
            row["dx"],
            row["dy"],
            head_width=2.5,
            head_length=2.0,
            fc=arrow_color,
            ec="black",
            alpha=alpha,
            linewidth=1.5,
            zorder=5,
        )

        # Mark start (green) and end (red)
        ax.scatter(
            row["start_x"],
            row["start_y"],
            c="green",
            s=80,
            edgecolors="black",
            linewidths=1.5,
            zorder=10,
            alpha=0.8,
        )
        ax.scatter(
            row["end_x"],
            row["end_y"],
            c="red",
            s=80,
            edgecolors="black",
            linewidths=1.5,
            zorder=10,
            alpha=0.8,
        )

    return ax
