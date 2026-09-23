from __future__ import annotations

from io import BytesIO

import pandas as pd


def load_csv(
    file_bytes: bytes,
    file_name: str,
) -> list[dict]:

    dataframe = pd.read_csv(
        BytesIO(file_bytes)
    )

    if dataframe.empty:
        return []

    # Convert each row into readable text.
    rows = []

    for index, row in dataframe.iterrows():

        parts = []

        for column in dataframe.columns:

            value = row[column]

            if pd.isna(value):
                continue

            parts.append(
                f"{column}: {value}"
            )

        if parts:

            rows.append(
                f"Row {index + 1}: "
                + " | ".join(parts)
            )

    if not rows:
        return []

    return [
        {
            "text": "\n".join(rows),
            "source_file": file_name,
            "page_number": None,
        }
    ]