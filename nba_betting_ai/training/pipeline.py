import numpy as np
import pandas as pd


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
    gameflow_idx_train = df_gameflow[gameflow_train_mask].groupby('game_id').sample(n=n, frac=frac).index
    gameflow_idx_test = df_gameflow[~gameflow_train_mask].groupby('game_id').sample(n=n, frac=frac).index

    return games_idx_train, games_idx_test, gameflow_idx_train, gameflow_idx_test
