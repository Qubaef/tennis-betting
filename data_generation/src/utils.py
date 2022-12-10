import os
import zipfile

import pandas as pd
import numpy as np

from typing import List


def unzip(zipPath: str, targetPath: str):
    with zipfile.ZipFile(zipPath, 'r') as zip_ref:
        zip_ref.extractall(targetPath)


def zip_files(filePaths: List[str], zipPath: str):
    # Zip given files into one zip file without any directory structure with regular compression
    with zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in filePaths:
            zip_file.write(file, os.path.basename(file))


def duration_to_minutes(duration: str):
    # Convert duration to minutes (eg. '01:02:00' -> 62)
    duration = duration.split(':')
    return int(duration[0]) * 60 + int(duration[1])


def date_to_timestamp(date: str):
    return float(pd.to_datetime(date).timestamp() / pd.Timestamp.max.timestamp())


def base_type(val):
    if isinstance(val, np.generic):
        return val.item()
    return val
