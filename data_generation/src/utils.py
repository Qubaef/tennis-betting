import os
import zipfile

import pandas as pd
import numpy as np

from typing import List


def unzip(zipPath: str, targetPath: str) -> None:
    with zipfile.ZipFile(zipPath, "r") as zip_ref:
        zip_ref.extractall(targetPath)


def zip_files(filePaths: List[str], zipPath: str) -> None:
    # Zip given files into one zip file without any directory structure with regular compression
    with zipfile.ZipFile(zipPath, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file in filePaths:
            zip_file.write(file, os.path.basename(file))


def duration_to_minutes(duration: str) -> int:
    # Convert duration to minutes (eg. '01:02:00' -> 62)
    timeParts = duration.split(":")
    return int(timeParts[0]) * 60 + int(timeParts[1])


def date_to_timestamp(date: str, min_date: str, max_date: str) -> float:
    return float(
        (pd.to_datetime(date) - pd.to_datetime(min_date)).total_seconds()
        / (pd.to_datetime(max_date) - pd.to_datetime(min_date)).total_seconds()
    )

def base_type(val):
    if isinstance(val, np.generic):
        return val.item()
    return val
