"""
Space analysis: Voronoi diagrams and space creation metrics
"""

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from src.utils import calculate_distance


def calculate_voronoi_areas(df, frame_number, pitch_length=105, pitch_width=68):
    """
    Calculate Voronoi diagram areas for one frame

    Args:
        df: Tracking DataFrame
        frame_number: Which frame to analyze
        pitch_length: Pitch length in meters
        pitch_width: Pitch width in meters

    Returns:
        Dict with player_id -> controlled_area (m²)
    """
    # Get all players in this frame
    frame_df = df[df["frame"] == frame_number].copy()

    if len(frame_df) < 4:
        return {}

    # Get positions
    points = frame_df[["x", "y"]].values
    player_ids = frame_df["player_id"].values

    # Calculate Voronoi
    try:
        vor = Voronoi(points)
    except Exception as e:
        return {}

    # Calculate areas using convex hull approximation
    # For each point, find its Voronoi region
    areas = {}

    for i, player_id in enumerate(player_ids):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]

        # Skip infinite regions
        if -1 in region or len(region) == 0:
            # Assign boundary area
            areas[player_id] = pitch_length * pitch_width / len(player_ids)
            continue

        # Get vertices of the region
        vertices = vor.vertices[region]

        # Calculate polygon area using Shoelace formula
        if len(vertices) >= 3:
            x = vertices[:, 0]
            y = vertices[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            # Cap at pitch size
            max_area = pitch_length * pitch_width
            areas[player_id] = min(area, max_area)
        else:
            areas[player_id] = pitch_length * pitch_width / len(player_ids)

    return areas


def measure_space_creation(df, player_id, start_frame, end_frame, teammate_ids):
    """
    Measure how much space a player created for teammates

    Args:
        df: Tracking DataFrame
        player_id: Player who made the run
        start_frame: Frame before the run
        end_frame: Frame after the run
        teammate_ids: List of teammate player IDs

    Returns:
        Dict with teammate_id -> space_gained (m²)
    """
    # Calculate Voronoi areas before and after
    areas_before = calculate_voronoi_areas(df, start_frame)
    areas_after = calculate_voronoi_areas(df, end_frame)

    if not areas_before or not areas_after:
        return {}

    # Measure space change for each teammate
    space_created = {}

    for tm_id in teammate_ids:
        if tm_id in areas_before and tm_id in areas_after:
            # Space gained = area after - area before
            gain = areas_after[tm_id] - areas_before[tm_id]

            if gain > 0:  # Only positive gains
                space_created[tm_id] = gain

    return space_created


def analyze_offball_runs(
    df,
    possession_df,
    home_player_ids,
    away_player_ids,
    velocity_threshold=5.0,
    window_frames=30,
):
    """
    Analyze off-ball runs - ONLY when own team has possession
    """
    from src.utils import detect_runs
    from tqdm import tqdm
    import pandas as pd

    # Create player_id to team mapping
    player_to_team = {}
    for pid in home_player_ids:
        player_to_team[pid] = "home"
    for pid in away_player_ids:
        player_to_team[pid] = "away"

    # Detect all runs
    runs = detect_runs(df, velocity_threshold)

    print(f"\nAnalyzing {len(runs):,} run frames (team possession filter)...")

    results = []

    for idx, run in tqdm(runs.iterrows(), total=len(runs), desc="Analyzing runs"):
        player_id = run["player_id"]
        frame = run["frame"]

        # Get runner's team
        runner_team = player_to_team.get(player_id)
        if runner_team is None:
            continue

        # Get possession info
        poss_frame = possession_df[possession_df["frame"] == frame]

        if len(poss_frame) == 0:
            continue

        poss_player_id = poss_frame.iloc[0]["player_id"]

        # Skip if NaN (no possession data)
        if pd.isna(poss_player_id):
            continue

        # Skip if runner has ball
        if poss_player_id == player_id:
            continue

        # Get possessor team
        poss_team = player_to_team.get(poss_player_id)

        if poss_team is None:
            continue

        # Only count if runner's team has possession
        if runner_team != poss_team:
            continue

        # Get teammates
        if runner_team == "home":
            teammates = [p for p in home_player_ids if p != player_id]
        else:
            teammates = [p for p in away_player_ids if p != player_id]

        if len(teammates) == 0:
            continue

        # Measure space creation
        start = frame
        end = frame + window_frames

        space_created = measure_space_creation(df, player_id, start, end, teammates)

        if space_created:
            total = sum(space_created.values())

            results.append(
                {
                    "frame": frame,
                    "player_id": player_id,
                    "team": runner_team,
                    "velocity": run["velocity"],
                    "space_created": total,
                    "teammates_benefited": len(space_created),
                    "is_detected": run["is_detected"],
                }
            )

    results_df = pd.DataFrame(results)

    print(f"✓ Analyzed {len(results_df)} off-ball runs (own team possession only)")

    return results_df


def group_runs_to_trajectories(runs_df):
    """
    Group consecutive frames into distinct runs with start/end points

    Args:
        runs_df: DataFrame from analyze_offball_runs with x, y coordinates

    Returns:
        DataFrame with one row per run containing start_x, start_y, end_x, end_y
    """
    import pandas as pd

    if len(runs_df) == 0:
        return pd.DataFrame()

    # Sort by player and frame
    runs_df = runs_df.sort_values(["player_id", "frame"]).copy()

    # Identify run groups (new run if gap > 10 frames = 1 second)
    runs_df["frame_diff"] = runs_df.groupby("player_id")["frame"].diff()
    runs_df["run_id"] = (runs_df["frame_diff"] > 10).cumsum()

    # Group and get start/end for each run
    trajectories = []

    for (player_id, run_id), group in runs_df.groupby(["player_id", "run_id"]):
        if len(group) < 3:  # Skip very short runs
            continue

        first = group.iloc[0]
        last = group.iloc[-1]

        trajectories.append(
            {
                "player_id": player_id,
                "team": first["team"],
                "start_frame": first["frame"],
                "end_frame": last["frame"],
                "duration_frames": len(group),
                "start_x": first["x"],
                "start_y": first["y"],
                "end_x": last["x"],
                "end_y": last["y"],
                "max_velocity": group["velocity"].max(),
                "avg_velocity": group["velocity"].mean(),
                "total_space_created": group["space_created"].sum(),
                "is_detected": group["is_detected"].mean(),
            }
        )

    return pd.DataFrame(trajectories)


def analyze_all_matches_normalized(matches, data_loader_func, velocity_threshold=8.0):
    """
    Analyze all matches with normalized attack direction (left to right)

    Args:
        matches: List of match dicts with 'id'
        data_loader_func: Function to load match data (load_match_data)
        velocity_threshold: Min velocity for runs (m/s)

    Returns:
        DataFrame with all trajectories, normalized to attack left->right
    """
    import pandas as pd
    from tqdm import tqdm

    all_trajectories = []

    print(f"\n=== ANALYZING ALL MATCHES (threshold: {velocity_threshold} m/s) ===\n")

    for match in tqdm(matches, desc="Processing matches"):
        match_id = match["id"]

        try:
            # Load match data
            data = data_loader_func(match_id)

            # Get home team side for each period
            home_team_side = data["metadata"].get("home_team_side", [])

            # Process both periods
            for period in [1, 2]:
                # Get tracking
                from src.data_loader import get_tracking_dataframe, get_possession_info
                from src.utils import calculate_velocity

                df = get_tracking_dataframe(data["tracking"], period=period)
                if len(df) == 0:
                    continue

                poss_df = get_possession_info(data["tracking"])

                # Get team player IDs from possession
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

                # Calculate velocity
                df_vel = calculate_velocity(df)

                # Analyze runs
                runs = analyze_offball_runs(
                    df_vel,
                    poss_df,
                    home_player_ids,
                    away_player_ids,
                    velocity_threshold=velocity_threshold,
                )

                if len(runs) == 0:
                    continue

                # Merge coordinates
                runs = runs.merge(
                    df_vel[["frame", "player_id", "x", "y"]],
                    on=["frame", "player_id"],
                    how="left",
                )

                # Group into trajectories
                traj = group_runs_to_trajectories(runs)

                if len(traj) == 0:
                    continue

                # NORMALIZE DIRECTION based on home_team_side
                # Period index: 0=first half, 1=second half
                period_idx = period - 1

                if period_idx < len(home_team_side):
                    home_direction = home_team_side[period_idx]

                    # Determine which team needs flipping
                    # If home attacks 'right_to_left', flip home team coordinates
                    # If home attacks 'left_to_right', flip away team coordinates

                    if home_direction == "right_to_left":
                        # Home team attacks right to left → need to flip
                        # Away team attacks left to right → keep as is
                        mask_home = traj["team"] == "home"
                        traj.loc[mask_home, "start_x"] *= -1
                        traj.loc[mask_home, "end_x"] *= -1
                    else:  # 'left_to_right'
                        # Home team attacks left to right → keep as is
                        # Away team attacks right to left → need to flip
                        mask_away = traj["team"] == "away"
                        traj.loc[mask_away, "start_x"] *= -1
                        traj.loc[mask_away, "end_x"] *= -1

                # Add match and period info
                traj["match_id"] = match_id
                traj["period"] = period

                all_trajectories.append(traj)

        except Exception as e:
            print(f"Error processing match {match_id}: {e}")
            continue

    if len(all_trajectories) == 0:
        print("No trajectories found!")
        return pd.DataFrame()

    # Combine all
    combined = pd.concat(all_trajectories, ignore_index=True)

    print(f"\n✓ Total runs across all matches: {len(combined)}")
    print(f"✓ Avg velocity: {combined['max_velocity'].mean():.2f} m/s")
    print(f"✓ Avg space created: {combined['total_space_created'].mean():.0f} m²")

    return combined
