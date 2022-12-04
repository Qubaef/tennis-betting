import os

# Use path relatively to the script path,
# because Pycharm sets directories differently than when running from command line
filepath = os.path.dirname(os.path.abspath(__file__))
rootpath = f'{filepath}/..'

ORG_DATASET_ZIP_PATH: str = f'{rootpath}/tennisDatasetOrg.zip'
ORG_DATASET_PATH: str = f'{rootpath}/data/org'
ORG_DATASET_MATCHES_CSV_PATH: str = f'{ORG_DATASET_PATH}/all_matches.csv'

OWN_DATASET_DIR: str = f'{rootpath}/data/own'
OWN_FULL_DATASET_PATH: str = f'{rootpath}/data/own/data.json'
OWN_TRAIN_DATASET_PATH: str = f'{OWN_DATASET_DIR}/train'
OWN_TEST_DATASET_PATH: str = f'{OWN_DATASET_DIR}/test'
OWN_VAL_DATASET_PATH: str = f'{OWN_DATASET_DIR}/val'

CLEAN_DATASET_PATH: str = f'{rootpath}/clean_dataset.csv'
