import os

# Use path relatively to the script path,
# because Pycharm sets directories differently than when running from command line
filepath = os.path.dirname(os.path.abspath(__file__))
rootpath = f'{filepath}/..'

ORG_DATASET_ZIP_PATH: str = f'{rootpath}/tennisDatasetOrg.zip'
ORG_DATASET_PATH: str = f'{rootpath}/data/org'

OWN_DATASET_PATH: str = f'{rootpath}/data/own'
OWN_TRAIN_DATASET_PATH: str = f'{OWN_DATASET_PATH}/train'
OWN_TEST_DATASET_PATH: str = f'{OWN_DATASET_PATH}/test'
OWN_VAL_DATASET_PATH: str = f'{OWN_DATASET_PATH}/val'
