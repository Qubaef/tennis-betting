import json
# import jsonpickle
import os
import shutil
import math
from tqdm import tqdm
from typing import List, Dict

import pandas as pd

import src.utils as utils
import src.paths as paths

RECENT_MATCHES_COUNT = 7
H2H_MATCHES_COUNT = 3


class MatchConditions:
    def __init__(self):
        self.tournamentId: str = ''
        self.tournamentCourtSurface: str = ''
        self.tournamentPrizeMoney: float = 0
        self.tournamentRound: int = 0

    def from_row(self, row: pd.Series):
        self.tournamentId = row['tournament']
        self.tournamentCourtSurface = row['court_surface']
        self.tournamentPrizeMoney = row['prize_money']
        self.tournamentRound = row['round_num']


class GameStats:
    def __init__(self):
        self.setsWon: float = 0
        self.setsLost: float = 0
        self.gamesWon: float = 0
        self.gamesLost: float = 0
        self.tiebreaksWon: float = 0
        self.tiebreaksLost: float = 0
        self.serveRating: float = 0
        self.aces: float = 0
        self.doubleFaults: float = 0
        self.firstServeMade: float = 0
        self.firstServeAttempted: float = 0
        self.firstServePointsMade: float = 0
        self.firstServePointsAttempted: float = 0
        self.secondServePointsMade: float = 0
        self.secondServePointsAttempted: float = 0
        self.breakPointsSaved: float = 0
        self.breakPointsAgainst: float = 0
        self.serviceGamesWon: float = 0
        self.returnRating: float = 0
        self.firstServeReturnPointsMade: float = 0
        self.firstServeReturnPointsAttempted: float = 0
        self.secondServeReturnPointsMade: float = 0
        self.secondServeReturnPointsAttempted: float = 0
        self.breakPointsMade: float = 0
        self.breakPointsAttempted: float = 0
        self.returnGamesPlayed: float = 0
        self.servicePointsWon: float = 0
        self.servicePointsAttempted: float = 0
        self.returnPointsWon: float = 0
        self.returnPointsAttempted: float = 0
        self.totalPointsWon: float = 0
        self.totalPoints: float = 0
        self.wonFirstSet: bool = False

    def from_row(self, row: pd.Series):
        self.setsWon = row['sets_won']
        self.setsLost = row['num_sets'] - row['sets_won']
        self.gamesWon = row['games_won']
        self.gamesLost = row['games_against']
        self.tiebreaksWon = row['tiebreaks_won']
        self.tiebreaksLost = row['tiebreaks_total'] - row['tiebreaks_won']
        self.serveRating = row['serve_rating']
        self.aces = row['aces']
        self.doubleFaults = row['double_faults']
        self.firstServeMade = row['first_serve_made']
        self.firstServeAttempted = row['first_serve_attempted']
        self.firstServePointsMade = row['first_serve_points_made']
        self.firstServePointsAttempted = row['first_serve_points_attempted']
        self.secondServePointsMade = row['second_serve_points_made']
        self.secondServePointsAttempted = row['second_serve_points_attempted']
        self.breakPointsSaved = row['break_points_saved']
        self.breakPointsAgainst = row['break_points_against']
        self.serviceGamesWon = row['service_games_won']
        self.returnRating = row['return_rating']
        self.firstServeReturnPointsMade = row['first_serve_return_points_made']
        self.firstServeReturnPointsAttempted = row['first_serve_return_points_attempted']
        self.secondServeReturnPointsMade = row['second_serve_return_points_made']
        self.secondServeReturnPointsAttempted = row['second_serve_return_points_attempted']
        self.breakPointsMade = row['break_points_made']
        self.breakPointsAttempted = row['break_points_attempted']
        self.returnGamesPlayed = row['return_games_played']
        self.servicePointsWon = row['service_points_won']
        self.servicePointsAttempted = row['service_points_attempted']
        self.returnPointsWon = row['return_points_won']
        self.returnPointsAttempted = row['return_points_attempted']
        self.totalPointsWon = row['total_points_won']
        self.totalPoints = row['total_points']
        self.wonFirstSet = row['won_first_set']

    def aggregate_from_table(self, table: pd.DataFrame):
        # Avg stats from table
        self.setsWon = table['sets_won'].mean()
        self.setsLost = table['num_sets'].mean() - table['sets_won'].mean()
        self.gamesWon = table['games_won'].mean()
        self.gamesLost = table['games_against'].mean()
        self.tiebreaksWon = table['tiebreaks_won'].mean()
        self.tiebreaksLost = table['tiebreaks_total'].mean() - table['tiebreaks_won'].mean()
        self.serveRating = table['serve_rating'].mean()
        self.aces = table['aces'].mean()
        self.doubleFaults = table['double_faults'].mean()
        self.firstServeMade = table['first_serve_made'].mean()
        self.firstServeAttempted = table['first_serve_attempted'].mean()
        self.firstServePointsMade = table['first_serve_points_made'].mean()
        self.firstServePointsAttempted = table['first_serve_points_attempted'].mean()
        self.secondServePointsMade = table['second_serve_points_made'].mean()
        self.secondServePointsAttempted = table['second_serve_points_attempted'].mean()
        self.breakPointsSaved = table['break_points_saved'].mean()
        self.breakPointsAgainst = table['break_points_against'].mean()
        self.serviceGamesWon = table['service_games_won'].mean()
        self.returnRating = table['return_rating'].mean()
        self.firstServeReturnPointsMade = table['first_serve_return_points_made'].mean()
        self.firstServeReturnPointsAttempted = table['first_serve_return_points_attempted'].mean()
        self.secondServeReturnPointsMade = table['second_serve_return_points_made'].mean()
        self.secondServeReturnPointsAttempted = table['second_serve_return_points_attempted'].mean()
        self.breakPointsMade = table['break_points_made'].mean()
        self.breakPointsAttempted = table['break_points_attempted'].mean()
        self.returnGamesPlayed = table['return_games_played'].mean()
        self.servicePointsWon = table['service_points_won'].mean()
        self.servicePointsAttempted = table['service_points_attempted'].mean()
        self.returnPointsWon = table['return_points_won'].mean()
        self.returnPointsAttempted = table['return_points_attempted'].mean()
        self.totalPointsWon = table['total_points_won'].mean()
        self.totalPoints = table['total_points'].mean()
        self.wonFirstSet = len(table[table['won_first_set'] == 't']) / len(table)

class MatchStats:

    # Sample data:
    # start_date, end_date,   location, court_surface,prize_money,currency,year, player_id,     player_name, opponent_id,   opponent_name, tournament,        round,                num_sets,sets_won,games_won,games_against,tiebreaks_won,tiebreaks_total,serve_rating,aces,double_faults,first_serve_made,first_serve_attempted,first_serve_points_made,first_serve_points_attempted,second_serve_points_made,second_serve_points_attempted,break_points_saved,break_points_against,service_games_won,return_rating,first_serve_return_points_made,first_serve_return_points_attempted,second_serve_return_points_made,second_serve_return_points_attempted,break_points_made,break_points_attempted,return_games_played,service_points_won,service_points_attempted,return_points_won,return_points_attempted,total_points_won,total_points,duration,player_victory,retirement,seed,won_first_set,doubles,masters,round_num,nation
    # 2012-06-11, 2012-06-17, Slovakia, Clay,         30000,      €,       2012, adrian-partl,  A. Partl,    andrej-martin, A. Martin,     kosice_challenger, 2nd Round Qualifying, 2,       0,       3,        12,           0,            0,              149,         0,   5,            27,              44,                   12,                     27,                          4,                       17,                           1,                 7,                   8,                198,          8,                             30,8,14,1,1,7,16,44,16,44,32,88,01:02:00,f,f,,f,f,100,1,Slovakia
    # 2012-06-11, 2012-06-17, Slovakia, Clay,         30000,      €,       2012, andrej-martin, A. Martin,   adrian-partl,  A. Partl,      kosice_challenger, 2nd Round Qualifying, 2,       2,       12,       3,            0,            0,              268,         0,   1,            30,              44,                   22,                     30,                          6,                       14,                           0,                 1,                   7,                293,          15,                            27,13,17,6,7,8,28,44,28,44,56,88,01:02:00,t,f,8,t,f,100,1,Slovakia
    def __init__(self):
        self.matchConditions: MatchConditions = MatchConditions()
        self.duration: int = 0
        self.won: bool = False
        self.gameStats: GameStats = GameStats()

    def from_row(self, row: pd.Series):
        self.matchConditions.from_row(row)
        self.duration = utils.duration_to_minutes(row['duration'])
        self.won = row['player_victory'] == 't'
        self.gameStats.from_row(row)


class PlayerStats:
    def __init__(self):
        self.playerMatchStory: List[MatchStats] = []
        self.playerWins: int = 0
        self.playerLosses: int = 0
        self.aggregatedStats: GameStats = GameStats()

        self.h2hWins: int = 0
        self.h2hStory: List[MatchStats] = []


class MatchData:
    def __init__(self):
        self.matchConditions: MatchConditions = MatchConditions()

        self.player1Stats: PlayerStats = PlayerStats()
        self.player2Stats: PlayerStats = PlayerStats()


def get_player_stats(player_match_history: pd.DataFrame, versus_player_id: str) -> PlayerStats:
    player_stats: PlayerStats = PlayerStats()

    # Count wins and losses (wins are player_victory == 't')
    player_stats.playerWins = player_match_history[player_match_history['player_victory'] == 't'].shape[0]
    player_stats.playerLosses = len(player_match_history) - player_stats.playerWins

    # Get RECENT_MATCHES_COUNT last matches from history
    player_matches_before: pd.DataFrame = player_match_history.tail(RECENT_MATCHES_COUNT)

    for _, row in player_matches_before.iterrows():
        player_stats.playerMatchStory.append(MatchStats())
        player_stats.playerMatchStory[-1].from_row(row)

    # Get aggregated stats
    player_stats.aggregatedStats.aggregate_from_table(player_match_history)

    # Get h2h matches
    h2h_matches_player: pd.DataFrame = player_match_history[(player_match_history['opponent_id'] == versus_player_id)]

    # Get number of wins for each player
    player_stats.h2hWins = h2h_matches_player[h2h_matches_player['player_victory'] == 't'].shape[0]

    # Get H2H_MATCHES_COUNT last matches between the two players
    for _, row in h2h_matches_player.tail(H2H_MATCHES_COUNT).iterrows():
        player_stats.h2hStory.append(MatchStats())
        player_stats.h2hStory[-1].from_row(row)

    return player_stats


def load_org_data(csv_path: str):
    matches = pd.read_csv(csv_path, sep=',')

    # Sort by date
    matches = matches.sort_values(by=['start_date'])
    matches.to_csv(paths.CLEAN_DATASET_PATH, index=False)

    # Create a list of tables, each containing matches of a single player (by player_id)
    players = matches['player_id'].unique()
    players_matches: Dict[str, pd.DataFrame] = {}
    for player in tqdm(players):
        player_matches = matches[matches['player_id'] == player]
        players_matches[player] = player_matches

    # For each match, collect last n matches of the player and last n matches of the opponent (to date)
    parsed_matches: List[MatchData] = []

    MAX_ITERS: int = 100

    # Take match to collect data for
    for index, match in tqdm(matches.iterrows(), total=matches.shape[0]):
        #### Parse player
        player_matches: pd.DataFrame = players_matches[match.player_id]

        # Find the match in the player's matches (identify by index)
        player_match_pos: int = player_matches.index.get_loc(index)

        # Get all matches before the match in question
        player_matches_before: pd.DataFrame = player_matches.iloc[:player_match_pos]

        # Generate player stats from matches played before
        player_stats: PlayerStats = get_player_stats(player_matches_before, match.opponent_id)

        #### Parse opponent
        opponent_matches: pd.DataFrame = players_matches[match.opponent_id]
        # Find the match in the opponent's matches
        # (identify by start_date of the tournament and player_id and opponent_id, which are inverted)
        opponent_match = opponent_matches.index[
            (opponent_matches['start_date'] == match.start_date) &
            (opponent_matches['player_id'] == match.opponent_id) &
            (opponent_matches['opponent_id'] == match.player_id)]

        if len(opponent_match) > 0:
            opponent_match_index = opponent_match[0]
        else:
            print('No match found for opponent. Skipping...')
            continue

        # Find the match in the opponent's matches (identify by index)
        opponent_match_pos: int = opponent_matches.index.get_loc(opponent_match_index)

        # Get all matches before the match in question
        opponent_matches_before: pd.DataFrame = opponent_matches.iloc[:opponent_match_pos]

        # Generate opponent stats from matches played before
        opponent_stats: PlayerStats = get_player_stats(opponent_matches_before, match.player_id)

        #### Fill the match data
        parsed_matches.append(MatchData())
        parsed_match: MatchData = parsed_matches[-1]

        parsed_match.matchConditions.from_row(match)

        parsed_match.player1Stats = player_stats
        parsed_match.player2Stats = opponent_stats

        # if MAX_ITERS > 0:
        #     MAX_ITERS -= 1
        # else:
        #     break

    # Serialize parsed matches to json
    # with open(paths.OWN_FULL_DATASET_PATH, 'w') as outfile:
    #     json.dump([ob.__dict__ for ob in parsed_matches], outfile, default=lambda o: o.__dict__, indent=4)

    return matches


def clean_dataset(matches_csv_path: str, target_matches_csv_path: str):
    matches = pd.read_csv(matches_csv_path, sep=',')

    # Iterate over all matches and remove all matches that not match the criteria:
    # - retirement = 'f'
    # - prize_money = is not nan
    # - any of the stats is not nan (generalized to: serve_rating is not nan)
    # - all doubles matches (doubles is 'f')
    # - duration is not nanf

    # Filter out all stats
    # matches = matches[matches['player_id'] == 'rafael-nadal']
    matches = matches[matches['prize_money'].notnull()]
    matches = matches[matches['retirement'] == 'f']
    matches = matches[matches['doubles'] == 'f']
    matches = matches[matches['serve_rating'].notnull()]
    matches = matches[matches['duration'].notnull()]

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

    if os.path.exists(paths.OWN_DATASET_DIR):
        shutil.rmtree(paths.OWN_DATASET_DIR)
        # assert False, f'Own dataset folder already exists at {paths.OWN_DATASET_PATH}.' \
        #               f' If you want to regenerate the dataset, please remove the folder manually.'

    # Unzip org dataset
    utils.unzip(paths.ORG_DATASET_ZIP_PATH, paths.ORG_DATASET_PATH)

    # TMP: Copy org dataset to own dataset - to be replaced with proper data generation
    shutil.copytree(paths.ORG_DATASET_PATH, paths.OWN_DATASET_DIR)

    # Clean the dataset
    # clean_dataset(paths.ORG_DATASET_MATCHES_CSV_PATH, paths.CLEAN_DATASET_PATH)

    # Load org dataset
    matches = load_org_data(paths.ORG_DATASET_MATCHES_CSV_PATH)


if __name__ == '__main__':
    generate_dataset()
