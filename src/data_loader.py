import pandas as pd
import json
import requests
from pathlib import Path
import tempfile

# Temporary cache directory
DATA_DIR = Path(tempfile.gettempdir()) / "analytics_cup_cache"
DATA_DIR.mkdir(exist_ok=True)

# GitHub Base URL
BASE_URL = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"


def load_matches_info(match_ids=None):
    """
    Convert match IDs to dict format.

    Args:
        match_ids (list, optional): List of match ID strings

    Returns:
        list: List of dicts [{'id': 'match_id'}, ...]
    """
    if match_ids is None:
        return []
    return [{"id": str(match_id)} for match_id in match_ids]


def load_match_data(match_id, verbose=False):
    """
    Load match data from SkillCorner API with local caching.

    Args:
        match_id (str): Match identifier
        verbose (bool): Print download progress

    Returns:
        dict: Contains 'metadata', 'tracking', 'events', 'phases'
    """
    match_id = str(match_id)

    # Metadata
    meta_file = DATA_DIR / f"{match_id}_meta.json"
    if not meta_file.exists():
        url = f"{BASE_URL}/matches/{match_id}/{match_id}_match.json"
        response = requests.get(url)
        response.raise_for_status()
        meta_file.write_text(response.text, encoding="utf-8")

    metadata = json.loads(meta_file.read_text(encoding="utf-8"))

    # Tracking (Git LFS)
    track_file = DATA_DIR / f"{match_id}_tracking.jsonl"
    if not track_file.exists():
        url = f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"
        response = requests.get(url)
        response.raise_for_status()
        track_file.write_bytes(response.content)

    tracking = [
        json.loads(line)
        for line in track_file.read_text(encoding="utf-8").strip().split("\n")
        if line
    ]

    # Events
    events_file = DATA_DIR / f"{match_id}_events.csv"
    if not events_file.exists():
        url = f"{BASE_URL}/matches/{match_id}/{match_id}_dynamic_events.csv"
        response = requests.get(url)
        response.raise_for_status()
        events_file.write_text(response.text, encoding="utf-8")

    events = pd.read_csv(events_file, low_memory=False)

    # Phases
    phases_file = DATA_DIR / f"{match_id}_phases.csv"
    if not phases_file.exists():
        url = f"{BASE_URL}/matches/{match_id}/{match_id}_phases_of_play.csv"
        response = requests.get(url)
        response.raise_for_status()
        phases_file.write_text(response.text, encoding="utf-8")

    phases = pd.read_csv(phases_file, low_memory=False)

    if verbose:
        print(
            f"Loaded match {match_id}: {len(tracking):,} frames, {len(events):,} events, {len(phases):,} phases"
        )

    return {
        "metadata": metadata,
        "tracking": tracking,
        "events": events,
        "phases": phases,
    }


def get_tracking_dataframe(tracking_data, period=None):
    """
    Convert nested tracking data to flat DataFrame.

    Args:
        tracking_data (list): Frame dicts from load_match_data()
        period (int, optional): Filter by period (1 or 2)

    Returns:
        pd.DataFrame: Columns: frame, timestamp, period, player_id, x, y, is_detected
    """
    rows = []

    for frame_data in tracking_data:
        if frame_data.get("period") is None:
            continue

        if period is not None and frame_data["period"] != period:
            continue

        frame_info = {
            "frame": frame_data["frame"],
            "timestamp": frame_data["timestamp"],
            "period": frame_data["period"],
        }

        for player in frame_data.get("player_data", []):
            rows.append(
                {
                    **frame_info,
                    "player_id": player["player_id"],
                    "x": player["x"],
                    "y": player["y"],
                    "is_detected": player["is_detected"],
                }
            )

    return pd.DataFrame(rows)


def get_ball_dataframe(tracking_data, period=None):
    """
    Extract ball positions from tracking data.

    Args:
        tracking_data (list): Frame dicts from load_match_data()
        period (int, optional): Filter by period (1 or 2)

    Returns:
        pd.DataFrame: Columns: frame, timestamp, period, x, y
    """
    rows = []

    for frame_data in tracking_data:
        if frame_data.get("period") is None:
            continue

        if period is not None and frame_data["period"] != period:
            continue

        ball = frame_data.get("ball_data")
        if ball:
            rows.append(
                {
                    "frame": frame_data["frame"],
                    "timestamp": frame_data["timestamp"],
                    "period": frame_data["period"],
                    "x": ball.get("x"),
                    "y": ball.get("y"),
                }
            )

    return pd.DataFrame(rows)


def get_possession_info(tracking_data):
    """
    Extract possession information from tracking data.

    Args:
        tracking_data (list): Frame dicts from load_match_data()

    Returns:
        pd.DataFrame: Columns: frame, timestamp, period, player_id, group
    """
    rows = []

    for frame_data in tracking_data:
        if frame_data.get("period") is None:
            continue

        poss = frame_data.get("possession")
        if poss:
            rows.append(
                {
                    "frame": frame_data["frame"],
                    "timestamp": frame_data["timestamp"],
                    "period": frame_data["period"],
                    "player_id": poss.get("player_id"),
                    "group": poss.get("group"),
                }
            )

    return pd.DataFrame(rows)
