"""Load and normalize TSMC2014 Foursquare check-in files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FOURSQUARE_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "dataset_tsmc2014"
)
CHECKIN_COLUMNS = (
    "user_id",
    "item_id",
    "venue_category_id",
    "venue_category_name",
    "latitude",
    "longitude",
    "timezone_offset_minutes",
    "utc_timestamp_raw",
)
TIMESTAMP_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def load_foursquare_checkins(
    city: str = "NYC",
    data_dir: Path | str = DEFAULT_FOURSQUARE_DATA_DIR,
) -> pd.DataFrame:
    """Load one TSMC2014 city file into a canonical interaction table.

    Args:
        city: Dataset city code, either ``NYC`` or ``TKY``.
        data_dir: Directory containing the original TSMC2014 text files.

    Returns:
        Check-ins with canonical ``user_id``, ``item_id``, ``timestamp``, and
        ``source_order`` fields plus the unmodified venue metadata.

    Raises:
        FileNotFoundError: If the requested raw file is absent.
        NotADirectoryError: If ``data_dir`` is not a directory.
        ValueError: If the city, schema, or timestamps are invalid.
    """
    city_code = city.upper()
    if city_code not in {"NYC", "TKY"}:
        raise ValueError("city must be 'NYC' or 'TKY'")
    directory = Path(data_dir)
    if not directory.exists():
        raise FileNotFoundError(
            f"Foursquare data directory does not exist: {directory.resolve()}"
        )
    if not directory.is_dir():
        raise NotADirectoryError(
            f"Foursquare data path is not a directory: {directory.resolve()}"
        )
    path = directory / f"dataset_TSMC2014_{city_code}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Foursquare check-in file does not exist: {path}")

    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=list(CHECKIN_COLUMNS),
        encoding="latin-1",
        dtype={
            "user_id": "int64",
            "item_id": "string",
            "venue_category_id": "string",
            "venue_category_name": "string",
            "latitude": "float64",
            "longitude": "float64",
            "timezone_offset_minutes": "int64",
            "utc_timestamp_raw": "string",
        },
    )
    if frame.empty:
        raise ValueError(f"Foursquare check-in file is empty: {path}")
    if tuple(frame.columns) != CHECKIN_COLUMNS:
        raise ValueError(
            f"Malformed Foursquare columns in {path}; expected {CHECKIN_COLUMNS}, "
            f"received {tuple(frame.columns)}"
        )
    if frame.isna().any().any():
        missing = frame.isna().sum()
        raise ValueError(
            f"Foursquare check-in file contains missing values: "
            f"{missing[missing.gt(0)].to_dict()}"
        )

    timestamps = pd.to_datetime(
        frame["utc_timestamp_raw"],
        format=TIMESTAMP_FORMAT,
        utc=True,
        errors="coerce",
    )
    invalid = int(timestamps.isna().sum())
    if invalid:
        raise ValueError(f"Foursquare check-in file has {invalid} invalid UTC timestamps")
    frame.insert(0, "source_order", np.arange(len(frame), dtype=np.int64))
    frame["timestamp"] = timestamps
    return frame


def make_order_experiment_frames(
    checkins: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Adapt canonical check-ins to the shared experiment's temporary schema.

    The shared runner currently names users, items, and interactions using its
    MovieLens-era column names. This adapter is intentionally isolated here;
    raw venue identifiers remain available in the returned mapping.

    Args:
        checkins: Canonical output from :func:`load_foursquare_checkins`.

    Returns:
        Compatibility interaction frame, item catalog, and integer-to-venue
        mapping. Repeated user-venue events and source-row order are preserved.
    """
    required = {"user_id", "item_id", "timestamp", "source_order"}
    missing = required - set(checkins.columns)
    if missing:
        raise ValueError(f"Missing canonical Foursquare columns: {sorted(missing)}")
    ordered_items = np.sort(checkins["item_id"].astype(str).unique())
    item_to_index = {item_id: index + 1 for index, item_id in enumerate(ordered_items)}
    ratings = pd.DataFrame(
        {
            "UserID": checkins["user_id"].to_numpy(np.int64),
            "MovieID": checkins["item_id"]
            .astype(str)
            .map(item_to_index)
            .to_numpy(np.int64),
            "Rating": np.ones(len(checkins), dtype=np.float32),
            "Timestamp": checkins["timestamp"]
            .map(lambda value: int(value.timestamp()))
            .to_numpy(np.int64),
        },
        index=checkins["source_order"].to_numpy(np.int64),
    ).sort_index()
    mapping = pd.DataFrame(
        {
            "MovieID": np.arange(1, len(ordered_items) + 1, dtype=np.int64),
            "item_id": ordered_items,
        }
    )
    items = mapping[["MovieID"]].copy()
    return ratings.reset_index(drop=True), items, mapping
