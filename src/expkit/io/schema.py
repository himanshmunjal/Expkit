"""The per-user metrics table contract.

Every data source must produce a frame that passes ``validate_user_metrics``.
This is the only coupling between the data layer and everything downstream,
so swapping BigQuery for Snowflake, a CSV, or a synthetic generator needs no
changes elsewhere.
"""

from __future__ import annotations

import pandas as pd

ID_COLUMN = "user_id"

COUNT_COLUMNS = (
    "pre_answers", "pre_accepted", "pre_comments",
    "post_answers", "post_accepted", "post_comments",
)
BINARY_COLUMNS = ("pre_active", "post_active")
OTHER_COLUMNS = ("tenure_days",)

REQUIRED_COLUMNS = (ID_COLUMN, *COUNT_COLUMNS, *BINARY_COLUMNS, *OTHER_COLUMNS)


class SchemaError(ValueError):
    pass


def validate_user_metrics(df: pd.DataFrame, required: tuple[str, ...] = REQUIRED_COLUMNS) -> pd.DataFrame:
    """Raise SchemaError on any contract violation; return the frame unchanged."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    if df.empty:
        raise SchemaError("frame is empty")
    if df[ID_COLUMN].duplicated().any():
        raise SchemaError("user_id is not unique")
    present = [c for c in required if c != ID_COLUMN]
    if df[present].isna().any().any():
        bad = df[present].columns[df[present].isna().any()].tolist()
        raise SchemaError(f"nulls in {bad}")
    for c in (c for c in COUNT_COLUMNS if c in required):
        if (df[c] < 0).any():
            raise SchemaError(f"{c} has negative values")
    for c in (c for c in BINARY_COLUMNS if c in required):
        if not df[c].isin((0, 1)).all():
            raise SchemaError(f"{c} must be 0/1")
    for period in ("pre", "post"):
        acc, ans = f"{period}_accepted", f"{period}_answers"
        if acc in df and ans in df and (df[acc] > df[ans]).any():
            raise SchemaError(f"{acc} exceeds {ans} for some users")
    return df
