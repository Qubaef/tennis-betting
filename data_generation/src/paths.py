import os

# Use path relatively to the script path,
# because Pycharm sets directories differently than when running from command line
filepath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.abspath(f"{filepath}/../../")

ORG_DATASET_ZIP_PATH: str = f"{rootpath}/tennisDatasetOrg.zip"
ORG_DATASET_DIR: str = f"{rootpath}/data/org"
ORG_DATASET_MATCHES_CSV_PATH: str = f"{ORG_DATASET_DIR}/all_matches.csv"
ORG_DATASET_BETS1_CSV_PATH: str = f"{ORG_DATASET_DIR}/betting_totals.csv"
ORG_DATASET_BETS2_CSV_PATH: str = f"{ORG_DATASET_DIR}/betting_spreads.csv"
ORG_DATASET_BETS3_CSV_PATH: str = f"{ORG_DATASET_DIR}/betting_moneyline.csv"

ORG_CLEAN_DATASET_DIR: str = f"{rootpath}/data/org_clean"
ORG_CLEAN_DATASET_ZIP_PATH: str = f"{rootpath}/tennisDatasetOrgClean.zip"
ORG_CLEAN_STATS_DATASET_PATH: str = f"{ORG_CLEAN_DATASET_DIR}/clean_stats_dataset.csv"
ORG_CLEAN_BETS_DATASET_PATH: str = f"{ORG_CLEAN_DATASET_DIR}/clean_bets_dataset.csv"

OWN_DATASET_DIR: str = f"{rootpath}/data/own"
OWN_FULL_DATASET_PATH: str = f"{rootpath}/data/own/data.csv"
OWN_TRAIN_DATASET_PATH: str = f"{OWN_DATASET_DIR}/train"
OWN_TEST_DATASET_PATH: str = f"{OWN_DATASET_DIR}/test"
OWN_VAL_DATASET_PATH: str = f"{OWN_DATASET_DIR}/val"
