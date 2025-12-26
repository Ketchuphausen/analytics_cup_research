"""
Visualization functions for football analytics
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np


def draw_pitch(
    ax=None, pitch_length=105, pitch_width=68, color="white", linecolor="black"
):
    """
    Draw a football pitch

    Args:
        ax: Matplotlib axis (creates new if None)
        pitch_length: Length in meters
        pitch_width: Width in meters
        color: Pitch color
        linecolor: Line color

    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Pitch outline
    ax.add_patch(
        patches.Rectangle(
            (-pitch_length / 2, -pitch_width / 2),
            pitch_length,
            pitch_width,
            facecolor=color,
            edgecolor=linecolor,
            linewidth=2,
        )
    )

    # Center line
    ax.plot([0, 0], [-pitch_width / 2, pitch_width / 2], color=linecolor, linewidth=2)

    # Center circle
    circle = plt.Circle((0, 0), 9.15, fill=False, color=linecolor, linewidth=2)
    ax.add_patch(circle)

    # Penalty boxes
    # Left
    ax.add_patch(
        patches.Rectangle(
            (-pitch_length / 2, -20.15),
            16.5,
            40.3,
            fill=False,
            edgecolor=linecolor,
            linewidth=2,
        )
    )
    # Right
    ax.add_patch(
        patches.Rectangle(
            (pitch_length / 2 - 16.5, -20.15),
            16.5,
            40.3,
            fill=False,
            edgecolor=linecolor,
            linewidth=2,
        )
    )

    # Goal boxes
    # Left
    ax.add_patch(
        patches.Rectangle(
            (-pitch_length / 2, -9.16),
            5.5,
            18.32,
            fill=False,
            edgecolor=linecolor,
            linewidth=2,
        )
    )
    # Right
    ax.add_patch(
        patches.Rectangle(
            (pitch_length / 2 - 5.5, -9.16),
            5.5,
            18.32,
            fill=False,
            edgecolor=linecolor,
            linewidth=2,
        )
    )

    # Set limits and aspect
    ax.set_xlim(-pitch_length / 2 - 5, pitch_length / 2 + 5)
    ax.set_ylim(-pitch_width / 2 - 5, pitch_width / 2 + 5)
    ax.set_aspect("equal")
    ax.axis("off")

    return ax


def plot_players(ax, df, frame_number, color_by="is_detected", show_ids=False):
    """
    Plot player positions for one frame

    Args:
        ax: Matplotlib axis
        df: Tracking DataFrame
        frame_number: Which frame to plot
        color_by: 'is_detected', 'player_id', or 'velocity'
        show_ids: Show player IDs as text

    Returns:
        Matplotlib axis
    """
    # Get frame data
    frame_df = df[df["frame"] == frame_number].copy()

    if len(frame_df) == 0:
        print(f"No data for frame {frame_number}")
        return ax

    # Color mapping
    if color_by == "is_detected":
        colors = [
            "red" if not detected else "green" for detected in frame_df["is_detected"]
        ]
        label = "Detected" if any(frame_df["is_detected"]) else None
    elif color_by == "velocity" and "velocity" in frame_df.columns:
        colors = frame_df["velocity"]
    else:
        colors = "blue"

    # Plot players
    scatter = ax.scatter(
        frame_df["x"],
        frame_df["y"],
        c=colors,
        s=200,
        alpha=0.7,
        edgecolors="black",
        linewidths=1.5,
        zorder=10,
    )

    # Add player IDs if requested
    if show_ids:
        for _, player in frame_df.iterrows():
            ax.text(
                player["x"],
                player["y"],
                str(player["player_id"])[-4:],  # Last 4 digits
                ha="center",
                va="center",
                fontsize=6,
                weight="bold",
            )

    return ax


def plot_voronoi(ax, df, frame_number, alpha=0.3):
    """
    Plot Voronoi diagram on pitch

    Args:
        ax: Matplotlib axis
        df: Tracking DataFrame
        frame_number: Which frame to plot
        alpha: Transparency of Voronoi regions

    Returns:
        Matplotlib axis
    """
    from scipy.spatial import Voronoi, voronoi_plot_2d

    # Get frame data
    frame_df = df[df["frame"] == frame_number].copy()

    if len(frame_df) < 4:
        print("Need at least 4 players for Voronoi")
        return ax

    # Get positions
    points = frame_df[["x", "y"]].values

    # Calculate Voronoi
    try:
        vor = Voronoi(points)
    except Exception as e:
        print(f"Voronoi failed: {e}")
        return ax

    # Plot Voronoi regions
    for region in vor.regions:
        if -1 not in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), alpha=alpha, edgecolor="black", linewidth=0.5)

    return ax


def plot_space_creation_heatmap(ax, runs_df, bins=20):
    """
    Plot heatmap showing where space was created

    Args:
        ax: Matplotlib axis
        runs_df: DataFrame from analyze_offball_runs() with x, y, space_created
        bins: Number of bins for heatmap

    Returns:
        Matplotlib axis
    """
    if "x" not in runs_df.columns or "y" not in runs_df.columns:
        print("Need x, y columns in runs_df")
        return ax, None

    if len(runs_df) == 0:
        print("No runs to plot")
        return ax, None

    # Create 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        runs_df["x"], runs_df["y"], bins=bins, weights=runs_df["space_created"]
    )

    # Normalize to avoid all-white
    heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap

    # Plot heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(
        heatmap.T,
        extent=extent,
        origin="lower",
        cmap="Reds",  # Changed to Reds
        alpha=0.7,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    # Add scatter points on top
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
