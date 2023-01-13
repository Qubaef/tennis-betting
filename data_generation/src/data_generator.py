import math
import os
import shutil
from tqdm import tqdm
from typing import List, Dict, Any

import pandas as pd

from data_generation.src import utils, paths

RECENT_MATCHES_COUNT = 7
H2H_MATCHES_COUNT = 3

MIN_DATE = "1990-01-01"
MAX_DATE = "2030-01-01"

# Sample data:
# start_date, end_date,   location, court_surface,prize_money,currency,year, player_id,     player_name, opponent_id,   opponent_name, tournament,        round,                num_sets,sets_won,games_won,games_against,tiebreaks_won,tiebreaks_total,serve_rating,aces,double_faults,first_serve_made,first_serve_attempted,first_serve_points_made,first_serve_points_attempted,second_serve_points_made,second_serve_points_attempted,break_points_saved,break_points_against,service_games_won,return_rating,first_serve_return_points_made,first_serve_return_points_attempted,second_serve_return_points_made,second_serve_return_points_attempted,break_points_made,break_points_attempted,return_games_played,service_points_won,service_points_attempted,return_points_won,return_points_attempted,total_points_won,total_points,duration,player_victory,retirement,seed,won_first_set,doubles,masters,round_num,nation # noqa: E501
# 2012-06-11, 2012-06-17, Slovakia, Clay,         30000,      €,       2012, adrian-partl,  A. Partl,    andrej-martin, A. Martin,     kosice_challenger, 2nd Round Qualifying, 2,       0,       3,        12,           0,            0,              149,         0,   5,            27,              44,                   12,                     27,                          4,                       17,                           1,                 7,                   8,                198,          8,                             30,8,14,1,1,7,16,44,16,44,32,88,01:02:00,f,f,,f,f,100,1,Slovakia # noqa: E501
# 2012-06-11, 2012-06-17, Slovakia, Clay,         30000,      €,       2012, andrej-martin, A. Martin,   adrian-partl,  A. Partl,      kosice_challenger, 2nd Round Qualifying, 2,       2,       12,       3,            0,            0,              268,         0,   1,            30,              44,                   22,                     30,                          6,                       14,                           0,                 1,                   7,                293,          15,                            27,13,17,6,7,8,28,44,28,44,56,88,01:02:00,t,f,8,t,f,100,1,Slovakia # noqa: E501

courtSurfaces: Dict[str, int] = {}
tournamentIds: Dict[str, int] = {}


def courtSurface_to_float(surface: str) -> float:
    if surface not in courtSurfaces:
        courtSurfaces[surface] = len(courtSurfaces)

    return courtSurfaces[surface]


def tournamentId_to_float(tournamentId: str) -> float:
    if tournamentId not in tournamentIds:
        tournamentIds[tournamentId] = len(tournamentIds)

    return tournamentIds[tournamentId]


def tournamentStart_to_float(start_date: str) -> float:
    return utils.date_to_timestamp(start_date, MIN_DATE, MAX_DATE)


class MatchConditions:
    def __init__(self):
        self.tournamentStart: float = 0
        self.tournamentId: float = 0
        self.tournamentCourtSurface: float = 0
        self.tournamentReputation: float = 0
        self.tournamentRound: float = 0

    def from_row(self, row: pd.Series) -> None:
        self.tournamentStart = tournamentStart_to_float(row["start_date"])
        self.tournamentId = tournamentId_to_float(row["tournament"])
        self.tournamentCourtSurface = courtSurface_to_float(row["court_surface"])

        # WA: Set convert types because of pandas stupid bullshit
        self.tournamentReputation = float(utils.base_type(row["masters"]))

        # Rounds are numbered from -2 to 7, where 7 is final - normalize to 0-9, where 0 is final
        self.tournamentRound = float(7 - utils.base_type(row["round_num"]))

    @staticmethod
    def to_csv_columns(prefix: str = "") -> List[str]:
        columns: List[str] = [
            f"{prefix}tournamentStart",
            f"{prefix}tournamentId",
            f"{prefix}tournamentCourtSurface",
            f"{prefix}tournamentReputation",
            f"{prefix}tournamentRound",
        ]

        return columns

    def to_csv_row(self) -> List[float]:
        return [
            self.tournamentStart,
            self.tournamentId,
            self.tournamentCourtSurface,
            self.tournamentReputation,
            self.tournamentRound,
        ]


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
        self.wonFirstSet: float = 0

    def from_row(self, row: pd.Series) -> None:
        self.setsWon = row["sets_won"]
        self.setsLost = row["num_sets"] - row["sets_won"]
        self.gamesWon = row["games_won"]
        self.gamesLost = row["games_against"]
        self.tiebreaksWon = row["tiebreaks_won"]
        self.tiebreaksLost = row["tiebreaks_total"] - row["tiebreaks_won"]
        self.serveRating = row["serve_rating"]
        self.aces = row["aces"]
        self.doubleFaults = row["double_faults"]
        self.firstServeMade = row["first_serve_made"]
        self.firstServeAttempted = row["first_serve_attempted"]
        self.firstServePointsMade = row["first_serve_points_made"]
        self.firstServePointsAttempted = row["first_serve_points_attempted"]
        self.secondServePointsMade = row["second_serve_points_made"]
        self.secondServePointsAttempted = row["second_serve_points_attempted"]
        self.breakPointsSaved = row["break_points_saved"]
        self.breakPointsAgainst = row["break_points_against"]
        self.serviceGamesWon = row["service_games_won"]
        self.returnRating = row["return_rating"]
        self.firstServeReturnPointsMade = row["first_serve_return_points_made"]
        self.firstServeReturnPointsAttempted = row[
            "first_serve_return_points_attempted"
        ]
        self.secondServeReturnPointsMade = row["second_serve_return_points_made"]
        self.secondServeReturnPointsAttempted = row[
            "second_serve_return_points_attempted"
        ]
        self.breakPointsMade = row["break_points_made"]
        self.breakPointsAttempted = row["break_points_attempted"]
        self.returnGamesPlayed = row["return_games_played"]
        self.servicePointsWon = row["service_points_won"]
        self.servicePointsAttempted = row["service_points_attempted"]
        self.returnPointsWon = row["return_points_won"]
        self.returnPointsAttempted = row["return_points_attempted"]
        self.totalPointsWon = row["total_points_won"]
        self.totalPoints = row["total_points"]
        self.wonFirstSet = float(row["won_first_set"] == "t")

    def aggregate_from_table(self, table: pd.DataFrame) -> None:
        # Avg stats from table
        self.setsWon = table["sets_won"].mean()
        self.setsLost = table["num_sets"].mean() - table["sets_won"].mean()
        self.gamesWon = table["games_won"].mean()
        self.gamesLost = table["games_against"].mean()
        self.tiebreaksWon = table["tiebreaks_won"].mean()
        self.tiebreaksLost = (
            table["tiebreaks_total"].mean() - table["tiebreaks_won"].mean()
        )
        self.serveRating = table["serve_rating"].mean()
        self.aces = table["aces"].mean()
        self.doubleFaults = table["double_faults"].mean()
        self.firstServeMade = table["first_serve_made"].mean()
        self.firstServeAttempted = table["first_serve_attempted"].mean()
        self.firstServePointsMade = table["first_serve_points_made"].mean()
        self.firstServePointsAttempted = table["first_serve_points_attempted"].mean()
        self.secondServePointsMade = table["second_serve_points_made"].mean()
        self.secondServePointsAttempted = table["second_serve_points_attempted"].mean()
        self.breakPointsSaved = table["break_points_saved"].mean()
        self.breakPointsAgainst = table["break_points_against"].mean()
        self.serviceGamesWon = table["service_games_won"].mean()
        self.returnRating = table["return_rating"].mean()
        self.firstServeReturnPointsMade = table["first_serve_return_points_made"].mean()
        self.firstServeReturnPointsAttempted = table[
            "first_serve_return_points_attempted"
        ].mean()
        self.secondServeReturnPointsMade = table[
            "second_serve_return_points_made"
        ].mean()
        self.secondServeReturnPointsAttempted = table[
            "second_serve_return_points_attempted"
        ].mean()
        self.breakPointsMade = table["break_points_made"].mean()
        self.breakPointsAttempted = table["break_points_attempted"].mean()
        self.returnGamesPlayed = table["return_games_played"].mean()
        self.servicePointsWon = table["service_points_won"].mean()
        self.servicePointsAttempted = table["service_points_attempted"].mean()
        self.returnPointsWon = table["return_points_won"].mean()
        self.returnPointsAttempted = table["return_points_attempted"].mean()
        self.totalPointsWon = table["total_points_won"].mean()
        self.totalPoints = table["total_points"].mean()
        self.wonFirstSet = len(table[table["won_first_set"] == "t"]) / len(table)

    @staticmethod
    def to_csv_columns(prefix: str = "") -> List[str]:
        columns: List[str] = [
            f"{prefix}setsWon",
            f"{prefix}setsLost",
            f"{prefix}gamesWon",
            f"{prefix}gamesLost",
            f"{prefix}tiebreaksWon",
            f"{prefix}tiebreaksLost",
            f"{prefix}serveRating",
            f"{prefix}aces",
            f"{prefix}doubleFaults",
            f"{prefix}firstServeMade",
            f"{prefix}firstServeAttempted",
            f"{prefix}firstServePointsMade",
            f"{prefix}firstServePointsAttempted",
            f"{prefix}secondServePointsMade",
            f"{prefix}secondServePointsAttempted",
            f"{prefix}breakPointsSaved",
            f"{prefix}breakPointsAgainst",
            f"{prefix}serviceGamesWon",
            f"{prefix}returnRating",
            f"{prefix}firstServeReturnPointsMade",
            f"{prefix}firstServeReturnPointsAttempted",
            f"{prefix}secondServeReturnPointsMade",
            f"{prefix}secondServeReturnPointsAttempted",
            f"{prefix}breakPointsMade",
            f"{prefix}breakPointsAttempted",
            f"{prefix}returnGamesPlayed",
            f"{prefix}servicePointsWon",
            f"{prefix}servicePointsAttempted",
            f"{prefix}returnPointsWon",
            f"{prefix}returnPointsAttempted",
            f"{prefix}totalPointsWon",
            f"{prefix}totalPoints",
            f"{prefix}wonFirstSet",
        ]
        return columns

    def to_csv_row(self) -> List[float]:
        return [
            self.setsWon,
            self.setsLost,
            self.gamesWon,
            self.gamesLost,
            self.tiebreaksWon,
            self.tiebreaksLost,
            self.serveRating,
            self.aces,
            self.doubleFaults,
            self.firstServeMade,
            self.firstServeAttempted,
            self.firstServePointsMade,
            self.firstServePointsAttempted,
            self.secondServePointsMade,
            self.secondServePointsAttempted,
            self.breakPointsSaved,
            self.breakPointsAgainst,
            self.serviceGamesWon,
            self.returnRating,
            self.firstServeReturnPointsMade,
            self.firstServeReturnPointsAttempted,
            self.secondServeReturnPointsMade,
            self.secondServeReturnPointsAttempted,
            self.breakPointsMade,
            self.breakPointsAttempted,
            self.returnGamesPlayed,
            self.servicePointsWon,
            self.servicePointsAttempted,
            self.returnPointsWon,
            self.returnPointsAttempted,
            self.totalPointsWon,
            self.totalPoints,
            self.wonFirstSet,
        ]


class MatchStats:
    def __init__(self):
        self.matchConditions: MatchConditions = MatchConditions()
        self.duration: float = 0
        self.won: float = False
        self.gameStats: GameStats = GameStats()

    def from_row(self, row: pd.Series) -> None:
        self.matchConditions.from_row(row)
        self.duration = float(utils.duration_to_minutes(row["duration"]))
        self.won = float(row["player_victory"] == "t")
        self.gameStats.from_row(row)

    @staticmethod
    def to_csv_columns(prefix: str = "") -> List[str]:
        columns: List[str] = [
            f"{prefix}duration",
            f"{prefix}won",
        ]

        columns.extend(MatchConditions.to_csv_columns(prefix))
        columns.extend(GameStats.to_csv_columns(prefix))

        return columns

    def to_csv_row(self) -> List[float]:
        row: List[Any] = [
            self.duration,
            self.won,
        ]

        row.extend(self.matchConditions.to_csv_row())
        row.extend(self.gameStats.to_csv_row())

        return row


class PlayerStats:
    def __init__(self):
        self.playerMatchStory: List[MatchStats] = []
        self.playerWins: float = 0
        self.playerLosses: float = 0
        self.winsPerSurface: float = 0
        self.lossesPerSurface: float = 0
        self.winsPerRound: float = 0
        self.lossesPerRound: float = 0
        self.winsPerImportance: float = 0
        self.lossesPerImportance: float = 0
        self.aggregatedStats: GameStats = GameStats()

        self.h2hWins: float = 0
        self.h2hLosses: float = 0
        self.h2hStory: List[MatchStats] = []

    @staticmethod
    def to_csv_columns(prefix: str = "") -> List[str]:
        columns: List[str] = [
            f"{prefix}totalWins",
            f"{prefix}totalLosses",
            f"{prefix}winsPerSurface",
            f"{prefix}lossesPerSurface",
            f"{prefix}winsPerRound",
            f"{prefix}lossesPerRound",
            f"{prefix}winsPerImportance",
            f"{prefix}lossesPerImportance",
            f"{prefix}h2hWins",
            f"{prefix}h2hLosses",
        ]

        for i in range(RECENT_MATCHES_COUNT):
            columns.extend(MatchStats.to_csv_columns(f"{prefix}prev_{i}_"))

        for i in range(H2H_MATCHES_COUNT):
            columns.extend(MatchStats.to_csv_columns(f"{prefix}h2h_{i}_"))

        columns.extend(GameStats.to_csv_columns(f"{prefix}aggStyle_"))

        return columns

    def to_csv_row(self) -> List[float]:
        row: List[float] = [
            self.playerWins,
            self.playerLosses,
            self.winsPerSurface,
            self.lossesPerSurface,
            self.winsPerRound,
            self.lossesPerRound,
            self.winsPerImportance,
            self.lossesPerImportance,
            self.h2hWins,
            self.h2hLosses,
        ]

        for i in range(RECENT_MATCHES_COUNT):
            if i < len(self.playerMatchStory):
                row.extend(self.playerMatchStory[i].to_csv_row())
            else:
                row.extend([0] * len(MatchStats.to_csv_columns()))

        for i in range(H2H_MATCHES_COUNT):
            if i < len(self.h2hStory):
                row.extend(self.h2hStory[i].to_csv_row())
            else:
                row.extend([0] * len(MatchStats.to_csv_columns()))

        row.extend(self.aggregatedStats.to_csv_row())

        return row


class MatchData:
    def __init__(self):
        self.matchConditions: MatchConditions = MatchConditions()

        self.startDate: str = ""
        self.player1: str = ""
        self.player2: str = ""
        self.odds1: float = 0
        self.odds2: float = 0
        self.winner: float = 0
        self.rank1: float = 0
        self.rank2: float = 0
        self.rankPts1: float = 0
        self.rankPts2: float = 0
        self.setsPlayed: float = 0
        self.gamesPlayed: float = 0

        self.player1Stats: PlayerStats = PlayerStats()
        self.player2Stats: PlayerStats = PlayerStats()

    @staticmethod
    def to_csv_columns() -> List[str]:
        columns: List[str] = [
            "startDate",
            "player1",
            "player2",
            "odds1",
            "odds2",
            "winner",
            "rank1",
            "rank2",
            "rankPts1",
            "rankPts2",
            "setsPlayed",
            "gamesPlayed",
        ]

        columns.extend(MatchConditions.to_csv_columns())
        columns.extend(PlayerStats.to_csv_columns("p1_"))
        columns.extend(PlayerStats.to_csv_columns("p2_"))

        return columns

    def to_csv_row(self) -> pd.Series:
        row: pd.Series = pd.Series(
            [
                self.startDate,
                self.player1,
                self.player2,
                self.odds1,
                self.odds2,
                self.winner,
                self.rank1,
                self.rank2,
                self.rankPts1,
                self.rankPts2,
                self.setsPlayed,
                self.gamesPlayed,
            ]
        )
        row = pd.concat([row, pd.Series(self.matchConditions.to_csv_row())])
        row = pd.concat([row, pd.Series(self.player1Stats.to_csv_row())])
        row = pd.concat([row, pd.Series(self.player2Stats.to_csv_row())])

        return row


def get_player_stats(
    player_match_history: pd.DataFrame, versus_player_id: str, surface: str, round: int, importance: int
) -> PlayerStats:
    player_stats: PlayerStats = PlayerStats()

    if len(player_match_history) == 0:
        return player_stats

    # Count wins and losses (wins are player_victory == 't')
    player_stats.playerWins = float(
        player_match_history[player_match_history["player_victory"] == "t"].shape[0]
    )
    player_stats.playerLosses = float(
        len(player_match_history) - player_stats.playerWins
    )

    # Count wins and losses per surface
    player_stats.winsPerSurface = float(
        player_match_history[
            (player_match_history["player_victory"] == "t") &
            (player_match_history["court_surface"] == surface)
        ].shape[0])
    player_stats.lossesPerSurface = float(
        player_match_history[
            (player_match_history["player_victory"] != "t") &
            (player_match_history["court_surface"] == surface)
        ].shape[0])

    # Count wins and losses per round
    player_stats.winsPerRound = float(
        player_match_history[
            (player_match_history["player_victory"] == "t") &
            (player_match_history["round_num"] == round)
        ].shape[0])
    player_stats.lossesPerRound = float(
        player_match_history[
            (player_match_history["player_victory"] != "t") &
            (player_match_history["round_num"] == round)
        ].shape[0])

    # Count wins and losses per importance
    player_stats.winsPerImportance = float(
        player_match_history[
            (player_match_history["player_victory"] == "t") &
            (player_match_history["masters"] == importance)
        ].shape[0])
    player_stats.lossesPerImportance = float(
        player_match_history[
            (player_match_history["player_victory"] != "t") &
            (player_match_history["masters"] == importance)
        ].shape[0])

    # Copy RECENT_MATCHES_COUNT last matches from history
    player_matches_before: pd.DataFrame = player_match_history.tail(
        RECENT_MATCHES_COUNT
    ).copy().iloc[::-1]

    # player_matches_before: pd.DataFrame = player_match_history.tail(
    #     RECENT_MATCHES_COUNT
    # )

    for _, row in player_matches_before.iterrows():
        player_stats.playerMatchStory.append(MatchStats())
        player_stats.playerMatchStory[-1].from_row(row)

    # Get aggregated stats
    player_stats.aggregatedStats.aggregate_from_table(player_match_history)

    # Get h2h matches
    h2h_matches: pd.DataFrame = player_match_history[
        (player_match_history["opponent_id"] == versus_player_id)
    ]

    # Get number of wins in h2h matches
    player_stats.h2hWins = float(
        h2h_matches[h2h_matches["player_victory"] == "t"].shape[0]
    )
    player_stats.h2hLosses = float(len(h2h_matches) - player_stats.h2hWins)

    # Get H2H_MATCHES_COUNT last matches from history
    h2h_matches = h2h_matches.tail(H2H_MATCHES_COUNT).copy().iloc[::-1]

    # Get H2H_MATCHES_COUNT last matches between the two players
    for _, row in h2h_matches.iterrows():
        player_stats.h2hStory.append(MatchStats())
        player_stats.h2hStory[-1].from_row(row)

    return player_stats


def generate_own_data() -> pd.DataFrame:
    matches = pd.read_csv(paths.ORG_CLEAN_STATS_DATASET_PATH, sep=",")
    odds = pd.read_csv(paths.ODDS_CSV_PATH, sep=",")

    # Sort by date
    matches = matches.sort_values(by=["start_date", "round_num"], ascending=[True, True])
    odds = odds.sort_values(by=["Date"])
    # matches.to_csv(paths.ORG_CLEAN_STATS_DATASET_PATH, index=False)

    # Verify if dates from matches table fit in the range (MIN_DATE, MAX_DATE)
    min_date: str = matches["start_date"].min()
    max_date: str = matches["start_date"].max()
    if min_date < MIN_DATE or max_date > MAX_DATE:
        assert False, "Dates from matches table are not in the range (MIN_DATE, MAX_DATE)"

    # Verify if dates from bets table fit in the range (MIN_DATE, MAX_DATE)
    min_date: str = odds["Date"].min()
    max_date: str = odds["Date"].max()
    if min_date < MIN_DATE or max_date > MAX_DATE:
        assert False, "Dates from odds table are not in the range (MIN_DATE, MAX_DATE)"

    # Create a list of tables, each containing matches of a single player (by player_id)
    players = matches["player_id"].unique()
    players_matches: Dict[str, pd.DataFrame] = {}

    print("Splitting matches by player...")
    for player in tqdm(players):
        player_matches = matches[matches["player_id"] == player]
        players_matches[player] = player_matches

    # Construct a dictionary of inverted player names to later use it with odds
    inverted_player_names: Dict[str, List[str]] = {}
    for player in players:
        split_name = player.split("-")
        split_name[0] = split_name[0].capitalize()
        split_name[1] = split_name[1].capitalize()

        inverted_player_name: str = split_name[1] + " " + split_name[0][0] + "."
        if inverted_player_name not in inverted_player_names:
            inverted_player_names[inverted_player_name] = [player]

        if inverted_player_name not in inverted_player_names:
            inverted_player_names[inverted_player_name].append(player)

    # For each match from bets, collect features and pack them up into a MatchData object
    parsed_matches: List[MatchData] = []

    print("Parsing matches...")

    MAX_ITERS = 100

    for index, match_odds in tqdm(odds.iterrows(), total=odds.shape[0]):
        # if MAX_ITERS > 0:
        #     MAX_ITERS -= 1
        # else:
        #     break

        if match_odds["Winner"] not in inverted_player_names:
            continue
        if match_odds["Loser"] not in inverted_player_names:
            continue

        if len(inverted_player_names[match_odds["Winner"]]) > 1:
            continue
        if len(inverted_player_names[match_odds["Loser"]]) > 1:
            continue

        date: str = match_odds["Date"]
        team1_name: str = inverted_player_names[match_odds["Winner"]][0]
        team2_name: str = inverted_player_names[match_odds["Loser"]][0]

        # Find the stats of the match
        # Find the match in the player stats
        if (
            team1_name not in players_matches or
                team2_name not in players_matches
        ):
            continue

        player1_matches: pd.DataFrame = players_matches[team1_name]
        player2_matches: pd.DataFrame = players_matches[team2_name]

        # Find the match in the player stats (date has to be in the range of the tournament)
        player1_match_id = player1_matches.index[
            (player1_matches["start_date"] <= date)
            & (player1_matches["end_date"] >= date)
            & (player1_matches["player_id"] == team1_name)
            & (player1_matches["opponent_id"] == team2_name)
        ]

        # If the match is not found (or multiple were found), skip
        if len(player1_match_id) != 1:
            continue

        player2_match_id = player2_matches.index[
            (player2_matches["start_date"] <= date)
            & (player2_matches["end_date"] >= date)
            & (player2_matches["player_id"] == team2_name)
            & (player2_matches["opponent_id"] == team1_name)
        ]

        if len(player1_match_id) != 1:
            continue

        player1_match = player1_matches.loc[player1_match_id[0]]
        player2_match = player2_matches.loc[player2_match_id[0]]

        player1_matches_before = player1_matches.loc[: player1_match_id[0]]
        player2_matches_before = player2_matches.loc[: player2_match_id[0]]

        # Exclude the match from the before matches (remove by index)
        player1_matches_before = player1_matches_before.drop(player1_match_id[0])
        player2_matches_before = player2_matches_before.drop(player2_match_id[0])

        # Generate player stats from match history
        player1_stats = get_player_stats(player1_matches_before,
            team2_name, player1_match["court_surface"], player1_match["round_num"], player1_match["masters"])
        player2_stats = get_player_stats(player2_matches_before,
            team1_name, player2_match["court_surface"], player2_match["round_num"], player2_match["masters"])

        # Fill the match data
        # Team 1 perspective
        parsed_matches.append(MatchData())
        parsed_match: MatchData = parsed_matches[-1]

        parsed_match.startDate = player1_match["start_date"]
        parsed_match.player1 = team1_name
        parsed_match.player2 = team2_name
        parsed_match.odds1 = match_odds["AvgW"]
        parsed_match.odds2 = match_odds["AvgL"]
        parsed_match.winner = 0 if (player1_match["player_victory"] == "t") else 1
        parsed_match.rank1 = match_odds["WRank"]
        parsed_match.rank2 = match_odds["LRank"]
        parsed_match.rankPts1 = match_odds["WPts"]
        parsed_match.rankPts2 = match_odds["LPts"]
        parsed_match.setsPlayed = float(player1_match["num_sets"])
        parsed_match.gamesPlayed = float(player1_match["games_won"] + player1_match["games_against"])

        parsed_match.matchConditions.from_row(player1_match)
        parsed_match.player1Stats = player1_stats
        parsed_match.player2Stats = player2_stats

        # Team 2 perspective
        parsed_matches.append(MatchData())
        parsed_match = parsed_matches[-1]

        parsed_match.startDate = player2_match["start_date"]
        parsed_match.player1 = team2_name
        parsed_match.player2 = team1_name
        parsed_match.odds1 = match_odds["AvgL"]
        parsed_match.odds2 = match_odds["AvgW"]
        parsed_match.winner = 0 if (player2_match["player_victory"] == "t") else 1
        parsed_match.rank1 = match_odds["LRank"]
        parsed_match.rank2 = match_odds["WRank"]
        parsed_match.rankPts1 = match_odds["LPts"]
        parsed_match.rankPts2 = match_odds["WPts"]
        parsed_match.setsPlayed = float(player2_match["num_sets"])
        parsed_match.gamesPlayed = float(player2_match["games_won"] + player2_match["games_against"])

        parsed_match.matchConditions.from_row(player2_match)
        parsed_match.player1Stats = player2_stats
        parsed_match.player2Stats = player1_stats

    # Print tournamentIds and courtSurfaces ids
    print("Tournaments:")
    for key, value in tournamentIds.items():
        print(f"\t{key}: {value}")

    print("Court surfaces:")
    for key, value in courtSurfaces.items():
        print(f"\t{key}: {value}")

    # Preprocess data to be stored in csv
    own_matches_list: List[pd.Series] = []

    print("Merging matches to a single dataframe...")

    for match in tqdm(parsed_matches):
        own_matches_list.append(match.to_csv_row())

    own_matches_df = pd.concat(own_matches_list, axis=1).T
    own_matches_df.columns = MatchData.to_csv_columns()

    # Store the data
    own_matches_df.to_csv(paths.OWN_FULL_DATASET_PATH, index=False)

    return matches


def clean_data():
    """
    Func to filter original dataset and keep only relevant data, from which the proper custom features can be extracted
    into own dataset.
    Org dataset from: https://www.kaggle.com/datasets/hakeem/atp-and-wta-tennis-data
    Paths from paths.py are used in this function.
    :return:
    """
    matches = pd.read_csv(paths.ORG_DATASET_MATCHES_CSV_PATH, sep=",")
    bets1 = pd.read_csv(paths.ORG_DATASET_BETS1_CSV_PATH, sep=",")
    bets2 = pd.read_csv(paths.ORG_DATASET_BETS2_CSV_PATH, sep=",")
    bets3 = pd.read_csv(paths.ORG_DATASET_BETS3_CSV_PATH, sep=",")

    # Iterate over all matches and remove all matches that don't match the following criteria:
    # - doesn't contain retirement (retirement == 'f')
    # - prize_money is not nan
    # - any of the stats is not nan (generalized to: serve_rating is not nan)
    # - is not a double match (doubles == 'f')
    # - duration is not nan
    # - match is in at least one of the bet datasets

    # Filter out all stats
    # matches = matches[matches['player_id'] == 'rafael-nadal']
    matches = matches[matches["prize_money"].notnull()]
    matches = matches[matches["retirement"] == "f"]
    matches = matches[matches["doubles"] == "f"]
    matches = matches[matches["serve_rating"].notnull()]
    matches = matches[matches["duration"].notnull()]

    # Match with betting data (match is identified by start_date, player_id, opponent_id)
    # Marge all bet datasets (they have the same columns)
    bets = pd.concat([bets1, bets2, bets3])
    bets = bets.drop_duplicates(subset=["start_date", "team1", "team2"], keep="first")

    # Leave only following columns: start_date,team1,team2,odds1,odds2
    bets = bets[["start_date", "team1", "team2", "odds1", "odds2"]]
    bets.sort_values(by=["start_date", "team1", "team2"])

    # Merge matches with bets by (start_date, player_id, opponent_id) with (start_date, team1, team2)
    # Join only the columns: odds1,odds2
    # matches = matches.merge(bets, how='inner', left_on=['start_date', 'player_id', 'opponent_id'],
    #     right_on=['start_date', 'team1', 'team2'])

    # Save the cleaned datasets
    matches.to_csv(paths.ORG_CLEAN_STATS_DATASET_PATH, index=False)
    bets.to_csv(paths.ORG_CLEAN_BETS_DATASET_PATH, index=False)

    # Zip the cleaned datasets
    utils.zip_files(
        [paths.ORG_CLEAN_STATS_DATASET_PATH, paths.ORG_CLEAN_BETS_DATASET_PATH],
        paths.ORG_CLEAN_DATASET_ZIP_PATH,
    )

    # Delete the unzipped datasets
    os.remove(paths.ORG_CLEAN_STATS_DATASET_PATH)
    os.remove(paths.ORG_CLEAN_BETS_DATASET_PATH)


def generate_clean_org_dataset():
    # Check if org zip dataset exists
    if not os.path.exists(paths.ORG_DATASET_ZIP_PATH):
        assert (
            False
        ), f"Org zip file not found at {os.path.normpath(paths.ORG_DATASET_ZIP_PATH)}"
    else:
        print(f"Org zip file found at {os.path.normpath(paths.ORG_DATASET_ZIP_PATH)}")

    # Check if the datasets already exist and notify the user
    # Don't delete them every time to save SSD read/write cycles
    if os.path.exists(paths.ORG_DATASET_DIR):
        # shutil.rmtree(paths.ORG_DATASET_DIR)
        assert False, (
            f"Org dataset folder already exists at {os.path.normpath(paths.ORG_DATASET_DIR)}."
            f" If you want to regenerate the dataset, please remove the folder manually."
        )

    if os.path.exists(paths.ORG_CLEAN_DATASET_ZIP_PATH):
        os.remove(paths.ORG_CLEAN_DATASET_ZIP_PATH)

    # Unzip org dataset
    utils.unzip(paths.ORG_DATASET_ZIP_PATH, paths.ORG_DATASET_DIR)

    # Clean the dataset
    clean_data()


def generate_own_dataset():
    # Check if org clean zip dataset exists
    if not os.path.exists(paths.ORG_CLEAN_DATASET_ZIP_PATH):
        assert (
            False
        ), f"Org clean zip file not found at {os.path.normpath(paths.ORG_CLEAN_DATASET_ZIP_PATH)}"
    else:
        print(
            f"Org clean zip file found at {os.path.normpath(paths.ORG_CLEAN_DATASET_ZIP_PATH)}"
        )

    # Check if odds zip file exists
    if not os.path.exists(paths.ODDS_ZIP_PATH):
        assert (
            False
        ), f"Odds zip file not found. Please run the script 'stats_scrapper.py' first."
    else:
        print(
            f"Odds csv file found at {os.path.normpath(paths.ODDS_ZIP_PATH)}"
        )

    # Check if the datasets already exist and notify the user
    # Don't delete them every time to save SSD read/write cycles
    if os.path.exists(paths.OWN_DATASET_DIR):
        # shutil.rmtree(paths.OWN_DATASET_DIR)
        assert False, (
            f"Own dataset folder already exists at {os.path.normpath(paths.OWN_DATASET_DIR)}."
            f" If you want to regenerate the dataset, please remove the folder manually."
        )

    if os.path.exists(paths.ORG_CLEAN_DATASET_DIR):
        shutil.rmtree(paths.ORG_CLEAN_DATASET_DIR)

    # Unzip org clean dataset
    utils.unzip(paths.ORG_CLEAN_DATASET_ZIP_PATH, paths.ORG_CLEAN_DATASET_DIR)

    # Unzip odds dataset
    utils.unzip(paths.ODDS_ZIP_PATH, paths.ODDS_DIR)

    # Create own dataset folder
    os.mkdir(paths.OWN_DATASET_DIR)

    # Generate own dataset
    generate_own_data()

def test_generated_data():
    if not os.path.exists(paths.ORG_CLEAN_STATS_DATASET_PATH):
        assert False, f"Clean dataset stats file not found at {os.path.normpath(paths.ORG_CLEAN_STATS_DATASET_PATH)}"

    if not os.path.exists(paths.ORG_CLEAN_BETS_DATASET_PATH):
        assert False, f"Clean dataset bets file not found at {os.path.normpath(paths.ORG_CLEAN_BETS_DATASET_PATH)}"

    if not os.path.exists(paths.OWN_FULL_DATASET_PATH):
        assert False, f"Own full dataset file not found at {os.path.normpath(paths.OWN_FULL_DATASET_PATH)}"

    matches = pd.read_csv(paths.ORG_CLEAN_STATS_DATASET_PATH, sep=",")
    bets = pd.read_csv(paths.ORG_CLEAN_BETS_DATASET_PATH, sep=",")
    generated_data = pd.read_csv(paths.OWN_FULL_DATASET_PATH, sep=",")

    # For each record in generated_data, check if it's start_date is larger than previous matches and h2h matches
    for index, row in generated_data.iterrows():
        start_date = row["tournamentStart"]
        player1 = row["player1"]
        player2 = row["player2"]

        prev_match: float = math.inf
        for i in range(RECENT_MATCHES_COUNT):
            curr_match: float = row[f"p1_prev_{i}_tournamentStart"]
            if curr_match > prev_match:
                assert False, f"Previous matches are not sorted for record {index}"
            if curr_match > start_date:
                assert False, f"Start date of generated data is smaller than previous matches for record {index}"
            prev_match = curr_match

        prev_match: float = math.inf
        for i in range(RECENT_MATCHES_COUNT):
            curr_match: float = row[f"p2_prev_{i}_tournamentStart"]
            if curr_match > prev_match:
                assert False, f"Previous matches are not sorted for record {index}"
            if curr_match > start_date:
                assert False, f"Start date of generated data is smaller than previous matches for record {index}"
            prev_match = curr_match

        prev_match: float = math.inf
        for i in range(H2H_MATCHES_COUNT):
            curr_match: float = row[f"p1_h2h_{i}_tournamentStart"]
            if curr_match > prev_match:
                assert False, f"H2h matches are not sorted for record {index}"
            if curr_match > start_date:
                assert False, f"Start date of generated data is smaller than previous matches for record {index}"
            prev_match = curr_match

        prev_match: float = math.inf
        for i in range(H2H_MATCHES_COUNT):
            curr_match: float = row[f"p2_h2h_{i}_tournamentStart"]
            if curr_match > prev_match:
                assert False, f"H2h matches are not sorted for record {index}"
            if curr_match > start_date:
                assert False, f"Start date of generated data is smaller than previous matches for record {index}"
            prev_match = curr_match


    # Create a list of tables, each containing matches of a single player (by player_id)
    players = matches["player_id"].unique()
    players_matches: Dict[str, pd.DataFrame] = {}

    print("Splitting matches by player...")

    for player in tqdm(players):
        player_matches = matches[matches["player_id"] == player]
        players_matches[player] = player_matches

if __name__ == "__main__":
    # generate_clean_org_dataset()
    generate_own_dataset()
    # test_generated_data()