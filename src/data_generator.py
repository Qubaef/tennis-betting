import os
import shutil
import math
from tqdm import tqdm
from typing import List

import pandas as pd

from src.utils import unzip
import src.paths as paths


class MatchData:

    # Sample data:
    # start_date, end_date,   location, court_surface,prize_money,currency,year, player_id,     player_name, opponent_id,   opponent_name, tournament,        round,                num_sets,sets_won,games_won,games_against,tiebreaks_won,tiebreaks_total,serve_rating,aces,double_faults,first_serve_made,first_serve_attempted,first_serve_points_made,first_serve_points_attempted,second_serve_points_made,second_serve_points_attempted,break_points_saved,break_points_against,service_games_won,return_rating,first_serve_return_points_made,first_serve_return_points_attempted,second_serve_return_points_made,second_serve_return_points_attempted,break_points_made,break_points_attempted,return_games_played,service_points_won,service_points_attempted,return_points_won,return_points_attempted,total_points_won,total_points,duration,player_victory,retirement,seed,won_first_set,doubles,masters,round_num,nation
    # 2012-06-11, 2012-06-17, Slovakia, Clay,         30000,      €,       2012, adrian-partl,  A. Partl,    andrej-martin, A. Martin,     kosice_challenger, 2nd Round Qualifying, 2,       0,       3,        12,           0,            0,              149,         0,   5,            27,              44,                   12,                     27,                          4,                       17,                           1,                 7,                   8,                198,          8,                             30,8,14,1,1,7,16,44,16,44,32,88,01:02:00,f,f,,f,f,100,1,Slovakia
    # 2012-06-11, 2012-06-17, Slovakia, Clay,         30000,      €,       2012, andrej-martin, A. Martin,   adrian-partl,  A. Partl,      kosice_challenger, 2nd Round Qualifying, 2,       2,       12,       3,            0,            0,              268,         0,   1,            30,              44,                   22,                     30,                          6,                       14,                           0,                 1,                   7,                293,          15,                            27,13,17,6,7,8,28,44,28,44,56,88,01:02:00,t,f,8,t,f,100,1,Slovakia
    def __init__(self):
        self.tournamentYear: int = 0
        self.tournamentId: str = ''
        self.tournamentName: str = ''
        self.tournamentStartDate: str = ''
        self.tournamentEndDate: str = ''
        self.tournamentCourtSurface: str = ''
        self.tournamentPrizeMoney: str = ''
        self.tournamentCurrency: str = ''
        self.tournamentRound: str = ''
        self.player1Id: str = ''
        self.player1Name: str = ''
        self.player2Id: str = ''
        self.player2Name: str = ''
        self.setsNum: int = 0


fields: List[str] = ['start_date', 'end_date', 'location', 'court_surface', 'prize_money', 'currency', 'year', \
    'player_id', 'player_name', 'opponent_id', 'opponent_name', 'tournament', 'round', 'num_sets', 'sets_won', \
    'games_won', 'games_against', 'tiebreaks_won', 'tiebreaks_total', 'serve_rating', 'aces', 'double_faults', \
    'first_serve_made', 'first_serve_attempted', 'first_serve_points_made', 'first_serve_points_attempted', \
    'second_serve_points_made', 'second_serve_points_attempted', 'break_points_saved', 'break_points_against', \
    'service_games_won', 'return_rating', 'first_serve_return_points_made', 'first_serve_return_points_attempted', \
    'second_serve_return_points_made', 'second_serve_return_points_attempted', 'break_points_made', \
    'break_points_attempted', 'return_games_played', 'service_points_won', 'service_points_attempted', \
    'return_points_won', 'return_points_attempted', 'total_points_won', 'total_points', 'duration', 'player_victory', \
    'retirement', 'seed', 'won_first_set', 'doubles', 'masters', 'round_num', 'nation']


def load_org_data(csv_path: str):
    matches = pd.read_csv(csv_path, sep=',')

    # Sort by date
    matches = matches.sort_values(by=['start_date'])
    matches.to_csv(paths.CLEAN_DATASET_PATH, index=False)

    # Create a list of tables, each containing matches of a single player (by player_id)
    players = matches['player_id'].unique()
    players_matches = []
    for player in tqdm(players):
        player_matches = matches[matches['player_id'] == player]
        players_matches.append(player_matches)

    return matches


def clean_dataset(matches_csv_path: str, target_matches_csv_path: str):
    matches = pd.read_csv(matches_csv_path, sep=',')

    # Iterate over all matches and remove all matches that not match the criteria:
    # - retirement = 'f'
    # - prize_money = is not nan
    # - any of the stats is not nan (generalized to: serve_rating is not nan)
    # - all doubles matches (doubles is 'f')

    # Filter out all stats
    # matches = matches[matches['player_id'] == 'rafael-nadal']
    matches = matches[matches['prize_money'].notnull()]
    matches = matches[matches['retirement'] == 'f']
    matches = matches[matches['doubles'] == 'f']
    matches = matches[matches['serve_rating'].notnull()]

    # Save the cleaned dataset
    matches.to_csv(target_matches_csv_path, index=False)


def generate_dataset():
    # Check if org zip dataset exists
    if not os.path.exists(paths.ORG_DATASET_ZIP_PATH):
        assert False, f'Org zip file not found at {paths.ORG_DATASET_ZIP_PATH}'
    else:
        print(f'Org zip file found at {paths.ORG_DATASET_ZIP_PATH}')

    # Check if the datasets already exist and notify the user
    # Don't delete them every time to save SSD read/write cycles
    if os.path.exists(paths.ORG_DATASET_PATH):
        shutil.rmtree(paths.ORG_DATASET_PATH)
        # assert False, f'Org dataset folder already exists at {paths.ORG_DATASET_PATH}.' \
        #               f' If you want to regenerate the dataset, please remove the folder manually.'

    if os.path.exists(paths.OWN_DATASET_PATH):
        shutil.rmtree(paths.OWN_DATASET_PATH)
        # assert False, f'Own dataset folder already exists at {paths.OWN_DATASET_PATH}.' \
        #               f' If you want to regenerate the dataset, please remove the folder manually.'

    # Unzip org dataset
    unzip(paths.ORG_DATASET_ZIP_PATH, paths.ORG_DATASET_PATH)

    # TMP: Copy org dataset to own dataset - to be replaced with proper data generation
    shutil.copytree(paths.ORG_DATASET_PATH, paths.OWN_DATASET_PATH)

    # Load org dataset
    matches = load_org_data(paths.ORG_DATASET_MATCHES_CSV_PATH)


if __name__ == '__main__':
    generate_dataset()
