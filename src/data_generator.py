import os
import shutil

from src.utils import unzip
import src.paths as paths


def generate_dataset():
    # Check if org zip dataset exists
    if not os.path.exists(paths.ORG_DATASET_ZIP_PATH):
        assert False, f'Org zip file not found at {paths.ORG_DATASET_ZIP_PATH}'
    else:
        print(f'Org zip file found at {paths.ORG_DATASET_ZIP_PATH}')

    # Check if the datasets already exist and notify the user
    # Don't delete them every time to save SSD read/write cycles
    if os.path.exists(paths.ORG_DATASET_PATH):
        assert False, f'Org dataset folder already exists at {paths.ORG_DATASET_PATH}.' \
                      f' If you want to regenerate the dataset, please remove the folder manually.'

    if os.path.exists(paths.OWN_DATASET_PATH):
        assert False, f'Own dataset folder already exists at {paths.OWN_DATASET_PATH}.' \
                      f' If you want to regenerate the dataset, please remove the folder manually.'

    # Unzip org dataset
    unzip(paths.ORG_DATASET_ZIP_PATH, paths.ORG_DATASET_PATH)

    # TMP: Copy org dataset to own dataset - to be replaced with proper data generation
    shutil.copytree(paths.ORG_DATASET_PATH, paths.OWN_DATASET_PATH)


if __name__ == '__main__':
    generate_dataset()
