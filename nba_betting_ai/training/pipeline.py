import numpy as np
import pandas as pd

from attrs import frozen

from nba_betting_ai.consts import game_info
from nba_betting_ai.data.processing import merge_game_data, prepare_game_data
from nba_betting_ai.data.storage import get_engine, load_teams, load_games, load_gameflow

def filter_nba_matchups(df_games: pd.DataFrame, df_teams: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out non-NBA matchups from the games DataFrame. This is done by removing games where
    one of the teams is not listed in official NBA teams list.

    Params:
        df_games (pd.DataFrame): DataFrame with games
        df_teams (pd.DataFrame): DataFrame with NBA teams
        
    Returns:
        pd.DataFrame: DataFrame with NBA matchups only
    """
    non_nba_matchups = df_games[~df_games['team_abbreviation'].isin(df_teams['abbreviation'])]['game_id'].unique()
    df_games = df_games[~df_games['game_id'].isin(non_nba_matchups)]
    return df_games


def train_test_split(
        df_games: pd.DataFrame,
        df_gameflow: pd.DataFrame,
        test_size: float,
        n: int | None = None,
        frac: float | None = None,
        seed: int | None = None
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split the games and game flow data into training and testing sets. Thes split is chronological,
    with test set containing the most recent games. The game flow data is sampled in one of two ways:
    either by number of samples or by fraction of the data. If both are provided, the number of samples
    will be used. If neither are provided, the entire data set will be used.

    Params:
        df_games (pd.DataFrame): DataFrame with games
        df_gameflow (pd.DataFrame): DataFrame with game flow data
        test_size (float): Proportion of the data to include in the test set
        n (int | None): Number of samples to include in the training set
        frac (float | None): Fraction of the data to include in the training set
        seed (int | None): Random seed for reproducibility

    Returns:
        tuple[pd.Series, pd.Series, pd.Series, pd.Series]: Indices for the training and testing sets,
            particularly: games_idx_train, games_idx_test, gameflow_idx_train, gameflow_idx_test.
    """
    limit_date = df_games['game_date'].quantile(1 - test_size)
    games_train_mask = df_games['game_date'] < limit_date
    games_idx_train = df_games[games_train_mask].index
    games_idx_test = df_games[~games_train_mask].index
    
    if seed is not None:
        np.random.seed(seed)
    if n is not None and frac is not None:
        frac = None
    if n is None and frac is None:
        frac = 1.0
    gameflow_train_mask = df_gameflow['game_id'].isin(df_games.loc[games_train_mask, 'game_id'])
    gameflow_idx_train = (
        df_gameflow[gameflow_train_mask]
        .groupby('game_id')
        .sample(n=n, frac=frac)
        .sort_values(
            by=['game_id', 'period', 'period_time_remaining', 'home_score', 'away_score'],
            ascending=[True, True, False, True, True]
        )
        .index
    )
    gameflow_idx_test = (
        df_gameflow[~gameflow_train_mask]
        .groupby('game_id')
        .sample(n=n, frac=frac)
        .sort_values(
            by=['game_id', 'period', 'period_time_remaining', 'home_score', 'away_score'],
            ascending=[True, True, False, True, True]
        )
        .index
    )

    return games_idx_train, games_idx_test, gameflow_idx_train, gameflow_idx_test



def insert_diff(df: pd.DataFrame, difference_cols: list[str], minuend_cols: str, subtrahend_cols: str) -> pd.DataFrame:
    for diff, minuend, substrahend in zip(difference_cols, minuend_cols, subtrahend_cols):
        df[diff] = df[minuend] - df[substrahend]
    return df

def augument_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = insert_diff(df,
        difference_cols=['score_diff', 'final_score_diff', 'home_season_pts_diff_avg', 'away_season_pts_diff_avg', 'home_last_5_pts_diff_avg', 'away_last_5_pts_diff_avg'],
        minuend_cols=['away_score', 'away_score_final', 'home_season_pts_for_avg', 'away_season_pts_for_avg', 'home_last_5_pts_for_avg', 'away_last_5_pts_for_avg'],
        subtrahend_cols=['home_score', 'home_score_final', 'home_season_pts_against_avg', 'away_season_pts_against_avg', 'home_last_5_pts_against_avg', 'away_last_5_pts_against_avg']
    )
    return df

@frozen
class DataSet:
    teams: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    data_summary: pd.DataFrame


def prepare_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the period and period time remaining columns with a single column representing 
    the time remaining in the game in minutes.

    Params:
        df (pd.DataFrame): DataFrame with game data

    Returns:
        pd.DataFrame: DataFrame with time remaining in the game in minutes    
    """
    df['time_remaining'] = (np.maximum(game_info.periods_regular - df['period'], 0)*game_info.period_min*game_info.sec_min + df['period_time_remaining'])
    df = df.drop(columns=['period', 'period_time_remaining'])
    return df


def prepare_data(seasons: int, seed: int, test_size: float, n: int | None, frac: float | None) -> DataSet:
    """
    Prepare the data for training and testing. The data is split into training and testing sets, with
    the testing set containing the most recent games. The game flow data is sampled in one of two ways:
    either by number of samples or by fraction of the data. If both are provided, the number of samples
    will be used. If neither are provided, the entire data set will be used.

    Params:
        seasons (int): Number of seasons to include in the data set
        seed (int): Random seed for reproducibility
        test_size (float): Proportion of the data to include in the test set
        n (int | None): Number of samples to include in the training set
        frac (float | None): Fraction of the data to include in the training set

    Returns:
        DataSet: Data set for training
    """
    engine = get_engine()
    df_teams = load_teams(engine)
    df_games = load_games(engine)
    seasons_available = sorted(df_games['season_id'].unique().tolist())
    # If possible, taking one extra season to fill statistics for start of the first used season:
    seasons_of_interest = seasons_available[-seasons - 1:] if len(seasons_available) > seasons else seasons_available 
    drop_games = df_games[df_games['season_id'] == seasons_of_interest[0]]['game_id'].unique().tolist() \
                        if len(seasons_available) > seasons \
                        else []
    df_games = filter_nba_matchups(df_games, df_teams)
    df_games = df_games[df_games['season_id'].isin(seasons_of_interest)]
    df_gameflow = load_gameflow(engine, game_id = list(df_games['game_id'].unique()))
    df_games = prepare_game_data(df_games, df_gameflow)
    df_games = df_games[~df_games['game_id'].isin(drop_games)]
    df_games = merge_game_data(df_games, df_gameflow)

    games_idx_train, games_idx_test, gameflow_idx_train, gameflow_idx_test = train_test_split(
        df_games, df_gameflow, test_size=test_size, n=n, seed=seed, frac=frac)
    df_train = df_games.loc[games_idx_train]
    df_test = df_games.loc[games_idx_test]
    df_gameflow_train = df_gameflow.loc[gameflow_idx_train]
    df_gameflow_test = df_gameflow.loc[gameflow_idx_test]

    X_train = pd.merge(df_train, df_gameflow_train, left_on='game_id', right_on='game_id')
    X_test = pd.merge(df_test, df_gameflow_test, left_on='game_id', right_on='game_id')

    X_train = prepare_time(X_train)
    X_test = prepare_time(X_test)

    for df in [X_train, X_test]:
        df = augument_dataset(df)

    data_summary = pd.DataFrame({
        'dataset': ['All', 'Train', 'Test'],
        'games': [len(df_games['game_id'].unique()), len(df_train['game_id'].unique()), len(df_test['game_id'].unique())],
        'score_changes': [df_gameflow.shape[0], df_gameflow_train.shape[0], df_gameflow_test.shape[0]],
        'final_datapoints': [len(X_train) + len(X_test), len(X_train), len(X_test)],
        'start': [df_games['game_date'].min().date(), df_train['game_date'].min().date(), df_test['game_date'].min().date()],
        'end': [df_games['game_date'].max().date(), df_train['game_date'].max().date(), df_test['game_date'].max().date()]
    })
    data_summary = data_summary.set_index('dataset')
    return DataSet(
        teams=df_teams,
        X_train=X_train,
        X_test=X_test,
        data_summary=data_summary
    )
