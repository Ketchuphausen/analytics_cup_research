"""
Space analysis using Voronoi diagrams to measure off-ball run effectiveness
"""

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from tqdm import tqdm


def calculate_voronoi_areas(df, frame_number, pitch_length=105, pitch_width=68):
    """
    Calculate controlled area per player using Voronoi tessellation.

    Args:
        df (pd.DataFrame): Tracking data
        frame_number (int): Frame to analyze
        pitch_length (float): Pitch length in meters
        pitch_width (float): Pitch width in meters

    Returns:
        dict: {player_id: controlled_area_m2}
    """
    frame_df = df[df["frame"] == frame_number]

    if len(frame_df) < 4:
        return {}

    points = frame_df[["x", "y"]].values
    player_ids = frame_df["player_id"].values

    try:
        vor = Voronoi(points)
    except:
        return {}

    areas = {}
    max_area = pitch_length * pitch_width
    avg_area = max_area / len(player_ids)

    for i, player_id in enumerate(player_ids):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]

        # Handle infinite/invalid regions
        if -1 in region or len(region) < 3:
            areas[player_id] = avg_area
            continue

        # Shoelace formula for polygon area
        vertices = vor.vertices[region]
        x, y = vertices[:, 0], vertices[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas[player_id] = min(area, max_area)

    return areas


def measure_space_creation(df, player_id, start_frame, end_frame, target_player_id):
    """
    Measure space created for a specific player by comparing Voronoi areas.

    Args:
        df (pd.DataFrame): Tracking data
        player_id (int): Player making the run
        start_frame (int): Frame before run
        end_frame (int): Frame after run
        target_player_id (int): Player to measure space creation for (ball carrier)

    Returns:
        float: Space gained in m² (0 if negative)
    """
    areas_before = calculate_voronoi_areas(df, start_frame)
    areas_after = calculate_voronoi_areas(df, end_frame)

    if not areas_before or not areas_after:
        return 0.0

    if target_player_id not in areas_before or target_player_id not in areas_after:
        return 0.0

    gain = areas_after[target_player_id] - areas_before[target_player_id]
    return max(0.0, gain)


def analyze_offball_runs(
    df,
    possession_df,
    home_player_ids,
    away_player_ids,
    velocity_threshold=5.0,
    window_frames=30,
):
    """
    Detect and analyze off-ball runs during own team possession.
    Measures space created ONLY for the ball carrier.

    Args:
        df (pd.DataFrame): Tracking data with velocity column
        possession_df (pd.DataFrame): Possession data
        home_player_ids (list): Home team player IDs
        away_player_ids (list): Away team player IDs
        velocity_threshold (float): Min velocity in m/s (default: 5.0)
        window_frames (int): Frames to measure impact (default: 30 = 3 sec)

    Returns:
        pd.DataFrame: Analyzed runs with space creation for ball carrier
    """
    from src.utils import detect_runs

    # Player to team mapping
    player_to_team = {}
    for pid in home_player_ids:
        player_to_team[pid] = "home"
    for pid in away_player_ids:
        player_to_team[pid] = "away"

    # Detect high-speed runs
    runs = detect_runs(df, velocity_threshold)

    print(f"\nAnalyzing {len(runs):,} run frames (team possession filter)...")

    results = []

    for idx, run in tqdm(runs.iterrows(), total=len(runs), desc="Analyzing runs"):
        player_id = run["player_id"]
        frame = run["frame"]

        runner_team = player_to_team.get(player_id)
        if runner_team is None:
            continue

        # Check possession
        poss_frame = possession_df[possession_df["frame"] == frame]
        if len(poss_frame) == 0:
            continue

        poss_player_id = poss_frame.iloc[0]["player_id"]
        if pd.isna(poss_player_id):
            continue

        # Skip if runner has ball
        if poss_player_id == player_id:
            continue

        # Check if own team has possession
        poss_team = player_to_team.get(poss_player_id)
        if poss_team is None or runner_team != poss_team:
            continue

        # Measure space created for ball carrier only
        space_created = measure_space_creation(
            df, player_id, frame, frame + window_frames, poss_player_id
        )

        if space_created > 0:
            results.append(
                {
                    "frame": frame,
                    "player_id": player_id,
                    "ball_carrier_id": int(poss_player_id),
                    "team": runner_team,
                    "velocity": run["velocity"],
                    "space_created": space_created,
                    "is_detected": run["is_detected"],
                }
            )

    results_df = pd.DataFrame(results)
    print(f"✓ Analyzed {len(results_df)} off-ball runs (space for ball carrier only)")

    return results_df


def group_runs_to_trajectories(runs_df):
    """
    Group consecutive run frames into trajectories with start/end positions.

    Args:
        runs_df (pd.DataFrame): Run frames with player_id, frame, x, y, velocity, space_created

    Returns:
        pd.DataFrame: Trajectories with start/end positions and space delta
    """
    if len(runs_df) == 0:
        return pd.DataFrame()

    # Check if run_id exists, if not create it
    if "run_id" not in runs_df.columns:
        runs_df = runs_df.sort_values(["player_id", "frame"])
        runs_df["frame_diff"] = runs_df.groupby("player_id")["frame"].diff()
        runs_df["is_new_run"] = (runs_df["frame_diff"] > 1) | (
            runs_df["frame_diff"].isna()
        )
        runs_df["run_id"] = runs_df.groupby("player_id")["is_new_run"].cumsum()
        runs_df["run_id"] = (
            runs_df["player_id"].astype(str) + "_" + runs_df["run_id"].astype(str)
        )

    trajectories = []

    for run_id in runs_df["run_id"].unique():
        run = runs_df[runs_df["run_id"] == run_id].sort_values("frame")

        if len(run) == 0:
            continue

        # Get first and last frame
        start = run.iloc[0]
        end = run.iloc[-1]

        # Calculate space created as DELTA (end - start)
        space_delta = end["space_created"] - start["space_created"]

        # Build trajectory dict (only include columns that exist)
        traj = {
            "run_id": run_id,
            "player_id": start["player_id"],
            "team": start["team"],
            "start_frame": start["frame"],
            "end_frame": end["frame"],
            "duration_frames": len(run),
            "start_x": start["x"],
            "start_y": start["y"],
            "end_x": end["x"],
            "end_y": end["y"],
            "max_velocity": run["velocity"].max(),
            "total_space_created": space_delta,
        }

        # Add match_id only if it exists
        if "match_id" in start.index:
            traj["match_id"] = start["match_id"]

        trajectories.append(traj)

    return pd.DataFrame(trajectories)


def analyze_all_matches_normalized(matches, data_loader_func, velocity_threshold=5.0):
    """
    Analyze all matches with normalized attack direction (left to right).

    Args:
        matches (list): Match dicts with 'id' key
        data_loader_func: Function to load match data
        velocity_threshold (float): Min velocity in m/s (default: 5.0)

    Returns:
        pd.DataFrame: All trajectories normalized to left-to-right attack
    """
    from src.data_loader import get_tracking_dataframe, get_possession_info
    from src.utils import calculate_velocity

    all_trajectories = []

    print(f"\n=== ANALYZING ALL MATCHES (threshold: {velocity_threshold} m/s) ===\n")

    for match in tqdm(matches, desc="Processing matches"):
        match_id = match["id"]

        try:
            data = data_loader_func(match_id)
            home_team_side = data["metadata"].get("home_team_side", [])

            for period in [1, 2]:
                df = get_tracking_dataframe(data["tracking"], period=period)
                if len(df) == 0:
                    continue

                poss_df = get_possession_info(data["tracking"])

                # Get team IDs from possession data
                home_player_ids = list(
                    poss_df[poss_df["group"] == "home team"]["player_id"]
                    .dropna()
                    .astype(int)
                    .unique()
                )
                away_player_ids = list(
                    poss_df[poss_df["group"] == "away team"]["player_id"]
                    .dropna()
                    .astype(int)
                    .unique()
                )

                if len(home_player_ids) == 0 or len(away_player_ids) == 0:
                    continue

                df_vel = calculate_velocity(df)
                runs = analyze_offball_runs(
                    df_vel,
                    poss_df,
                    home_player_ids,
                    away_player_ids,
                    velocity_threshold,
                )

                if len(runs) == 0:
                    continue

                runs = runs.merge(
                    df_vel[["frame", "player_id", "x", "y"]],
                    on=["frame", "player_id"],
                    how="left",
                )
                traj = group_runs_to_trajectories(runs)

                if len(traj) == 0:
                    continue

                # Normalize attack direction
                period_idx = period - 1
                if period_idx < len(home_team_side):
                    home_direction = home_team_side[period_idx]

                    if home_direction == "right_to_left":
                        mask_home = traj["team"] == "home"
                        traj.loc[mask_home, ["start_x", "end_x"]] *= -1
                    else:
                        mask_away = traj["team"] == "away"
                        traj.loc[mask_away, ["start_x", "end_x"]] *= -1

                traj["match_id"] = match_id
                traj["period"] = period
                all_trajectories.append(traj)

        except Exception as e:
            print(f"Error processing match {match_id}: {e}")
            continue

    if len(all_trajectories) == 0:
        return pd.DataFrame()

    combined = pd.concat(all_trajectories, ignore_index=True)

    print(f"\n✓ Total runs: {len(combined)}")
    print(f"✓ Avg velocity: {combined['max_velocity'].mean():.2f} m/s")
    print(f"✓ Avg space created: {combined['total_space_created'].mean():.0f} m²")

    return combined
