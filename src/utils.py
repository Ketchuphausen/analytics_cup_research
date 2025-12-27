import numpy as np
import pandas as pd


def calculate_distance(x1, y1, x2, y2):
    """
    Calculate Euclidean distance between two points.

    Args:
        x1, y1: First point coordinates (scalar or array)
        x2, y2: Second point coordinates (scalar or array)

    Returns:
        float or np.array: Distance in meters
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_velocity(df, player_id=None):
    """
    Calculate frame-to-frame velocity from position changes.

    Args:
        df (pd.DataFrame): Tracking data with columns: frame, timestamp, player_id, x, y
        player_id (int, optional): Calculate for specific player only

    Returns:
        pd.DataFrame: Original DataFrame with added 'velocity' column (m/s)
    """
    df = df.copy()

    if player_id is not None:
        df = df[df["player_id"] == player_id].copy()

    df = df.sort_values(["player_id", "frame"]).reset_index(drop=True)

    # Calculate distance and time delta
    df["x_diff"] = df.groupby("player_id")["x"].diff()
    df["y_diff"] = df.groupby("player_id")["y"].diff()
    df["distance"] = np.sqrt(df["x_diff"] ** 2 + df["y_diff"] ** 2)

    # Time delta in seconds
    df["time_delta"] = pd.to_timedelta(df["timestamp"]).dt.total_seconds().diff()

    # Velocity
    df["velocity"] = df["distance"] / df["time_delta"]
    df["velocity"] = df["velocity"].fillna(0)

    # Cleanup temporary columns
    df = df.drop(["x_diff", "y_diff", "distance", "time_delta"], axis=1)

    return df


def detect_runs(df, velocity_threshold=5.0, min_duration=3.0):
    """
    Filter for high-speed runs lasting at least min_duration seconds.

    Args:
        df (pd.DataFrame): Tracking data with 'velocity' column
        velocity_threshold (float): Minimum speed in m/s (default: 5.0)
        min_duration (float): Minimum duration in seconds (default: 3.0)

    Returns:
        pd.DataFrame: Filtered to sustained runs

    Raises:
        ValueError: If 'velocity' column missing
    """
    if "velocity" not in df.columns:
        raise ValueError(
            "DataFrame must have 'velocity' column. Run calculate_velocity() first."
        )

    # Filter for high velocity
    high_speed = df[df["velocity"] >= velocity_threshold].copy()

    # Group consecutive frames per player
    high_speed = high_speed.sort_values(["player_id", "frame"]).reset_index(drop=True)
    high_speed["frame_gap"] = high_speed.groupby("player_id")["frame"].diff()
    high_speed["run_id"] = (high_speed["frame_gap"] > 1).cumsum()

    # Calculate duration of each run
    run_durations = (
        high_speed.groupby(["player_id", "run_id"])
        .agg({"timestamp": ["first", "last"]})
        .reset_index()
    )

    run_durations.columns = ["player_id", "run_id", "start_time", "end_time"]
    run_durations["duration"] = (
        pd.to_timedelta(run_durations["end_time"]).dt.total_seconds()
        - pd.to_timedelta(run_durations["start_time"]).dt.total_seconds()
    )

    # Filter runs >= min_duration
    valid_runs = run_durations[run_durations["duration"] >= min_duration][
        ["player_id", "run_id"]
    ]

    # Keep only frames from valid runs
    runs = high_speed.merge(valid_runs, on=["player_id", "run_id"], how="inner")
    runs = runs.drop(["frame_gap", "run_id"], axis=1)

    print(
        f"✓ Detected {len(runs):,} frames with velocity >= {velocity_threshold} m/s and duration >= {min_duration}s"
    )
    print(f"✓ From {runs['player_id'].nunique()} unique players")

    return runs
