from __future__ import annotations

from io import BytesIO

import pandas as pd


def load_xlsx(
    file_bytes: bytes,
    file_name: str,
) -> list[dict]:

    excel_file = pd.ExcelFile(
        BytesIO(file_bytes)
    )

    documents = []

    for sheet_name in excel_file.sheet_names:

        dataframe = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
        )

        if dataframe.empty:
            continue

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
            continue

        text = (
            f"Excel Sheet: {sheet_name}\n\n"
            + "\n".join(rows)
        )

        documents.append(
            {
                "text": text,
                "source_file": file_name,
                "page_number": None,
                "sheet_name": sheet_name,
            }
        )

    return documents