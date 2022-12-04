import zipfile


def unzip(zipPath: str, targetPath: str):
    with zipfile.ZipFile(zipPath, 'r') as zip_ref:
        zip_ref.extractall(targetPath)


def duration_to_minutes(duration: str):
    # Convert duration to minutes (eg. '01:02:00' -> 62)
    duration = duration.split(':')
    return int(duration[0]) * 60 + int(duration[1])
