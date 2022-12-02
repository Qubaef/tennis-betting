import zipfile


def unzip(zipPath: str, targetPath: str):
    with zipfile.ZipFile(zipPath, 'r') as zip_ref:
        zip_ref.extractall(targetPath)