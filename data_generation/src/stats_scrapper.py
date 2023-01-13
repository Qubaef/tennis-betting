import os
from typing import List
import pandas as pd

import requests
from tqdm import tqdm

from data_generation.src import paths, utils


def download_odds(years: List[str]):
    """
    Download odds from the web and save them as a csv file.
    Works only for data since 2005, because Elo points were added since then.
    """

    # Remove old odds
    if os.path.exists(paths.ODDS_ZIP_PATH):
        os.remove(paths.ODDS_ZIP_PATH)

    MIN_YEAR = 2005
    MAX_YEAR = 2020

    page_url: str = "http://www.tennis-data.co.uk/"
    columns: List[str] = ["Date", "Winner", "Loser", "Court", "Best of", "WRank", "LRank", "WPts", "LPts"]
    brookers: List[str] = ["CB", "GB", "IW", "SB", "B365", "B&W", "PS", "EX", "UB", "LB"]

    main_df = None

    # Odds and results are stored in .xlsl/xls files: http://www.tennis-data.co.uk/<year>/<year>.<ext>

    # Validate if year is a valid year
    for year in years:
            assert year.isdigit() and MIN_YEAR <= int(year) <= MAX_YEAR, f"Year {year} is not a valid year"

    for year in tqdm(years):
        # Download the file
        file_url_xls: str = f"{page_url}{year}/{year}.xls"
        file_url_xlsx: str = f"{page_url}{year}/{year}.xlsx"

        r_xls = requests.get(file_url_xls, allow_redirects=True)
        r_xlsx = requests.get(file_url_xlsx, allow_redirects=True)

        # Check if request was successful
        if r_xls.status_code != 200 and r_xlsx.status_code != 200:
            assert f"No file for year {year} found"

        # Retrieve dataframe from the file
        if r_xls.status_code == 200:
            current_df = pd.read_excel(file_url_xls, sheet_name=year)
        else:
            current_df = pd.read_excel(file_url_xlsx, sheet_name=year)

        # Determine from which brookers the odds are available for the given year
        used_brookers = brookers.copy()
        winners: List[str] = []
        losers: List[str] = []
        for brooker in brookers:
            brooker_win_odds = f"{brooker}W"
            brooker_lose_odds = f"{brooker}L"

            if brooker_win_odds not in current_df.columns or brooker_lose_odds not in current_df.columns:
                used_brookers.remove(brooker)
            else:
                winners.append(brooker_win_odds)
                losers.append(brooker_lose_odds)

        # Create a new dataframe with only the columns we need
        current_df = current_df[columns + winners + losers]

        # Calculate max, min, and average odds for each match winner and lose
        current_df[f"MaxW"] = current_df[winners].max(axis=1)
        current_df[f"MinW"] = current_df[winners].min(axis=1)
        current_df[f"AvgW"] = current_df[winners].mean(axis=1)
        current_df[f"MaxL"] = current_df[losers].max(axis=1)
        current_df[f"MinL"] = current_df[losers].min(axis=1)
        current_df[f"AvgL"] = current_df[losers].mean(axis=1)

        # Drop the columns with odds for each brooker
        current_df.drop(winners + losers, axis=1, inplace=True)

        # Drop rows which contain NaN in any column
        current_df.dropna(inplace=True)

        # Save the file
        # current_df.to_csv(f"{paths.ODDS_DIR}/{year}.csv", index=False)

        # Add current_df to main_df
        if main_df is None:
            main_df = current_df
        else:
            main_df = pd.concat([main_df, current_df])

    # Save the file
    main_df.to_csv("odds.csv", index=False)

    # Zip the file to ODDS_ZIP_PATH
    utils.zip_files(["odds.csv"], paths.ODDS_ZIP_PATH)
    # Remove the unzipped file
    os.remove("odds.csv")

if __name__ == "__main__":
    years: List[str] = [str(year) for year in range(2005, 2019)]
    download_odds(years)