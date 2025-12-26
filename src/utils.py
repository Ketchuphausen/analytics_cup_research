"""
Simple utility functions for calculations
"""

import numpy as np
import pandas as pd


def calculate_distance(x1, y1, x2, y2):
    """
    Calculate Euclidean distance between two points

    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates

    Returns:
        Distance in meters
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_velocity(df, player_id=None):
    """
    Calculate velocity (speed) from position changes

    Args:
        df: Tracking DataFrame with columns: frame, timestamp, player_id, x, y
        player_id: Optional - calculate for specific player only

    Returns:
        DataFrame with added 'velocity' column (m/s)
    """
    df = df.copy()

    if player_id is not None:
        # Filter for specific player
        df = df[df["player_id"] == player_id].copy()

    # Sort by player and frame
    df = df.sort_values(["player_id", "frame"]).reset_index(drop=True)

    # Calculate distance moved
    df["x_prev"] = df.groupby("player_id")["x"].shift(1)
    df["y_prev"] = df.groupby("player_id")["y"].shift(1)
    df["time_prev"] = df.groupby("player_id")["timestamp"].shift(1)

    # Distance
    df["distance"] = calculate_distance(df["x_prev"], df["y_prev"], df["x"], df["y"])

    # Time delta (convert timestamp to seconds if needed)
    df["time_delta"] = (
        pd.to_timedelta(df["timestamp"]).dt.total_seconds()
        - pd.to_timedelta(df["time_prev"]).dt.total_seconds()
    )

    # Velocity = distance / time
    df["velocity"] = df["distance"] / df["time_delta"]

    # Clean up
    df = df.drop(["x_prev", "y_prev", "time_prev", "distance", "time_delta"], axis=1)

    # First frame has no velocity
    df["velocity"] = df["velocity"].fillna(0)

    return df


def detect_runs(df, velocity_threshold=5.0, min_duration=1.0):
    """
    Detect off-ball runs (high-speed movement without ball)

    Args:
        df: Tracking DataFrame with velocity column
        velocity_threshold: Minimum speed to count as "running" (m/s)
        min_duration: Minimum duration to count as a "run" (seconds)

    Returns:
        DataFrame filtered to only running frames
    """
    # Must have velocity column
    if "velocity" not in df.columns:
        raise ValueError(
            "DataFrame must have 'velocity' column. Use calculate_velocity() first."
        )

    # Filter for high velocity
    runs = df[df["velocity"] >= velocity_threshold].copy()

    print(f"✓ Detected {len(runs):,} frames with velocity >= {velocity_threshold} m/s")
    print(f"✓ From {runs['player_id'].nunique()} unique players")

    return runs
