import pandas as pd
import json
import requests
from pathlib import Path
import tempfile  # NEU

# Temporäres Cache-Verzeichnis (wird automatisch aufgeräumt)
DATA_DIR = Path(tempfile.gettempdir()) / "analytics_cup_cache"
DATA_DIR.mkdir(exist_ok=True)

# GitHub Base URL
BASE_URL = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"

# Hard-coded Match IDs (alle 10 verfügbaren A-League Matches)
MATCH_IDS = [
    "2017461",
    "1996435",
    "1886347",
    "1899585",
    "1925299",
    "1953632",
    "2006229",
    "2011166",
    "2013725",
    "2015213",
]


def load_matches_info():
    """Get list of available matches"""
    return [{"id": mid} for mid in MATCH_IDS]


def load_match_data(match_id):
    """
    Load complete match data

    Returns dict with:
    - metadata: match info, lineups, pitch size
    - tracking: list of frames with player positions
    - events: DataFrame with passes, shots, etc.
    - phases: DataFrame with attack phases
    """
    # 1. Metadata
    meta_file = DATA_DIR / f"{match_id}_meta.json"

    if not meta_file.exists():
        print(f"Lade Metadata für Match {match_id}...")
        meta_url = f"{BASE_URL}/matches/{match_id}/{match_id}_match.json"
        response = requests.get(meta_url)

        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"✓ Metadata gespeichert")
    else:
        print(f"✓ Metadata aus Cache")

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 2. Tracking Data
    track_file = DATA_DIR / f"{match_id}_tracking.jsonl"

    if not track_file.exists():
        print(f"Lade Tracking Data für Match {match_id}...")
        track_url = f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"
        response = requests.get(track_url)

        print(f"  Download: {len(response.content):,} bytes")

        with open(track_file, "wb") as f:
            f.write(response.content)
        print(f"✓ Tracking gespeichert")
    else:
        print(f"✓ Tracking aus Cache")

    tracking = []
    with open(track_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tracking.append(json.loads(line))

    print(f"✓ {len(tracking):,} Tracking Frames geladen")

    # 3. Events
    events_file = DATA_DIR / f"{match_id}_events.csv"

    if not events_file.exists():
        print(f"Lade Events für Match {match_id}...")
        events_url = f"{BASE_URL}/matches/{match_id}/{match_id}_dynamic_events.csv"
        response = requests.get(events_url)

        with open(events_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"✓ Events gespeichert")
    else:
        print(f"✓ Events aus Cache")

    events = pd.read_csv(events_file)
    print(f"✓ {len(events):,} Events geladen")

    # 4. Phases
    phases_file = DATA_DIR / f"{match_id}_phases.csv"

    if not phases_file.exists():
        print(f"Lade Phases für Match {match_id}...")
        phases_url = f"{BASE_URL}/matches/{match_id}/{match_id}_phases_of_play.csv"
        response = requests.get(phases_url)

        with open(phases_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"✓ Phases gespeichert")
    else:
        print(f"✓ Phases aus Cache")

    phases = pd.read_csv(phases_file)
    print(f"✓ {len(phases):,} Phases geladen")

    return {
        "metadata": metadata,
        "tracking": tracking,
        "events": events,
        "phases": phases,
    }


def get_tracking_dataframe(tracking_data, period=None):
    """
    Convert tracking from nested dict to flat DataFrame

    Args:
        tracking_data: list of frame dicts from load_match_data()
        period: None (both halves), 1 (first half), or 2 (second half)

    Returns:
        DataFrame with columns: frame, timestamp, period, player_id, x, y, is_detected
    """
    rows = []

    for frame_data in tracking_data:
        # Skip frames without period (warmup etc.)
        if frame_data.get("period") is None:
            continue

        # Filter by period if specified
        if period and frame_data["period"] != period:
            continue

        frame = frame_data["frame"]
        timestamp = frame_data["timestamp"]
        per = frame_data["period"]

        # Extract each player
        for player in frame_data.get("player_data", []):
            rows.append(
                {
                    "frame": frame,
                    "timestamp": timestamp,
                    "period": per,
                    "player_id": player["player_id"],
                    "x": player["x"],
                    "y": player["y"],
                    "is_detected": player["is_detected"],
                }
            )

    return pd.DataFrame(rows)


def get_ball_dataframe(tracking_data, period=None):
    """
    Extract ball positions from tracking data

    Args:
        tracking_data: list of frame dicts
        period: None (both halves), 1 (first half), or 2 (second half)

    Returns:
        DataFrame with ball x, y positions per frame
    """
    rows = []

    for frame_data in tracking_data:
        # Skip frames without period
        if frame_data.get("period") is None:
            continue

        # Filter by period
        if period and frame_data["period"] != period:
            continue

        # Extract ball if exists
        if "ball_data" in frame_data and frame_data["ball_data"]:
            ball = frame_data["ball_data"]
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
    Extract who has possession at each frame

    Args:
        tracking_data: list of frame dicts

    Returns:
        DataFrame with possession info per frame
    """
    rows = []

    for frame_data in tracking_data:
        # Skip frames without period
        if frame_data.get("period") is None:
            continue

        # Extract possession if exists
        if "possession" in frame_data and frame_data["possession"]:
            poss = frame_data["possession"]
            rows.append(
                {
                    "frame": frame_data["frame"],
                    "timestamp": frame_data["timestamp"],
                    "period": frame_data["period"],
                    "player_id": poss.get("player_id"),
                    "group": poss.get("group"),  # 'home' or 'away'
                }
            )

    return pd.DataFrame(rows)
