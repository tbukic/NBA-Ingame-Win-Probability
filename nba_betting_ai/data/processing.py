import numpy as np
import pandas as pd


def fill_na_with_last_season(df_games: pd.DataFrame, filling_in_principle: dict[str, str]) -> pd.DataFrame:
    """
    Fill missing values in the games data with the last season's data. Last datapoint from the last season is used
    to fill in the missing values. If the last season's data is missing, the missing values are filled in with 0.

    Params:
        df_games (pd.DataFrame): DataFrame with games
        filling_in_principle (dict[str, str]): Dictionary with the columns to fill in and the filling in principle
    
    Returns:
        pd.DataFrame: DataFrame with filled in missing values
    """

    df_last_season = (
        df_games
        .groupby(['team_id', 'season_id'], as_index=False)
        .agg(filling_in_principle)
    )

    df_last_season = df_last_season.sort_values(['team_id', 'season_id'])

    filled_in_keys = list(filling_in_principle.keys())
    temporary_keys = add_prefix(filled_in_keys, 'tmp_prev', return_type='list')

    df_last_season[temporary_keys] = (
        df_last_season
        .groupby('team_id')[filled_in_keys]
        .shift(1)
    )

    df_games = df_games.merge(
        df_last_season.drop(columns=filled_in_keys),
        on=['team_id', 'season_id'],
        how='left'
    )

    for original_key, temporary_key in zip(filled_in_keys, temporary_keys):
        df_games[original_key] = (
            df_games[original_key]
            .fillna(df_games[temporary_key])
            .fillna(0)
        )
    df_games = df_games.drop(columns=temporary_keys)
    return df_games


def fill_na(df_games: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the games data. A part of the data is filled in with the last season's data,
    and the rest is filled in with 0.

    Params:
        df_games (pd.DataFrame): DataFrame with games

    Returns:
        pd.DataFrame: DataFrame with filled in missing values
    """
    filling_in_principle = {
        'season_wins': 'last',
        'last_5_wins': 'last',
        'last_5_pts_for_avg' :'last',
        'last_5_pts_for_total': 'last',
        'last_5_pts_against_avg': 'last',
        'last_5_pts_against_total': 'last',
        'season_pts_for_avg': 'last',
        'season_wins_avg': 'last',
        'season_pts_against_avg': 'last',
    }
    df_games = fill_na_with_last_season(df_games, filling_in_principle)
    fill_w_zeros = ['season_pts_for', 'season_pts_against']
    df_games[fill_w_zeros] = (
        df_games[fill_w_zeros]
        .fillna(0)
    )
    return df_games


def prepare_game_data(df_games: pd.DataFrame, df_gameflow: pd.DataFrame, groups: str='game') -> pd.DataFrame:
    """
    Merges the game data with results and team form.

    Params:
        df_games (pd.DataFrame): DataFrame with games
        df_gameflow (pd.DataFrame): DataFrame with game flow data
        groups (str): Unit for which the results will be reported. Either 'period' or a 'game'.

    Returns:
        pd.DataFrame: DataFrame with game data
    """
    if not groups in ['period', 'game']:
        raise ValueError('Values for  must be either "period" or "game"')
    grouping_columns = ['game_id'] if groups == 'game' else ['game_id', 'period']
    drop_columns = ['period', 'period_time_remaining']  if groups == 'game' else ['period_time_remaining']
    df_results = (
            df_gameflow[df_gameflow['game_id'].isin(df_games['game_id'].unique())]
            .sort_values(
                by=['period', 'period_time_remaining'],
                ascending=[True, False]
            )
            .groupby(grouping_columns)
            .tail(1)
            .drop(
                columns=drop_columns
            )
    )
    df_games = pd.merge(df_games, df_results, left_on='game_id', right_on='game_id')
    df_games['game_date'] = pd.to_datetime(df_games['game_date'])
            
    df_games['points_for'] = np.where(
        df_games['wl'] == 'W',
        df_games[['home_score', 'away_score']].max(axis=1),
        df_games[['home_score', 'away_score']].min(axis=1)
    )
    df_games['points_against'] = np.where(
        df_games['wl'] == 'W',
        df_games[['home_score', 'away_score']].min(axis=1),
        df_games[['home_score', 'away_score']].max(axis=1)
    )
    df_games = df_games.drop(columns=['home_score', 'away_score'])
            
    df_games['win'] = (df_games['wl'] == 'W').astype(int)
            
    df_games = df_games.sort_values(['team_id', 'game_date'])
        
    df_games[['lag_win', 'lag_points_for', 'lag_points_against']] = (
        df_games
        .groupby('team_id')[['win', 'points_for', 'points_against']]
        .shift(1)
    )
    last_5 = df_games.groupby('team_id').rolling(window=5, min_periods=1).agg({
        'lag_win': 'sum',
        'lag_points_for': ['mean', 'sum'],
        'lag_points_against': ['mean', 'sum']
    })
    last_5.columns = ['last_5_wins', 'last_5_pts_for_avg', 'last_5_pts_for_total', 'last_5_pts_against_avg', 'last_5_pts_against_total']

    df_games[['lag_win', 'lag_points_for', 'lag_points_against']] = (
        df_games
        .groupby(['team_id', 'season_id'], dropna=False)[['win', 'points_for', 'points_against']]
        .shift(1)
    )
    df_games[['season_wins','season_pts_for','season_pts_against']] = (
        df_games
        .groupby(['team_id','season_id'])[['lag_win','lag_points_for','lag_points_against']]
        .cumsum()
    )
    df_games['season_games'] = (
        df_games
        .groupby(['team_id','season_id'])
        .cumcount()
    )

    df_games = df_games.drop(columns=['lag_win', 'lag_points_for', 'lag_points_against'])

    df_games = pd.merge(
        df_games.reset_index(drop=True),
        last_5.reset_index().drop('level_1', axis=1),
        left_index=True,
        right_index=True,
    ).drop(columns=['team_id_y']).rename(columns={'team_id_x': 'team_id'}).sort_values(['game_date'])
    
    df_games['season_wins_avg'] = df_games['season_wins'] / df_games['season_games']
    df_games['season_pts_for_avg'] = df_games['season_pts_for'] / df_games['season_games']
    df_games['season_pts_against_avg'] = df_games['season_pts_against'] / df_games['season_games']
    
    df_games = fill_na(df_games)

    return df_games.reset_index(drop=True)

def add_prefix(to_list: list[str], prefix: str, return_type: str = 'dict') -> dict[str, str]:
    """
    Add a prefix to each element in the list.

    Params:
        to_list (list[str]): List of elements
        prefix (str): Prefix
        return_type (str): Type of the return value. Either 'dict' or 'list'

    Returns:
        dict[str, str]: Dictionary with the elements and the prefix

    Raises:
        ValueError: If return_type is not 'dict' or 'list'
    """
    with_prefix = [
        f'{prefix}_{elem}'
        for elem in to_list
    ]
    if return_type == 'dict':
        return dict(zip(to_list, with_prefix))
    if return_type == 'list':
        return with_prefix
    raise ValueError('return_type must be either "dict" or "list"')

cols_team = [
    'team_id',
    'team_abbreviation',
    'season_wins',
    'season_pts_for',
    'season_pts_against',
    'season_games',
    'season_wins_avg',
    'season_pts_for_avg',
    'season_pts_against_avg',
    'last_5_wins',
    'last_5_pts_for_avg',
    'last_5_pts_for_total',
    'last_5_pts_against_avg',
    'last_5_pts_against_total',
]
"""
A list of columns related to each teams statistics.
"""
cols_base = ['game_id', 'win', 'game_date',]
"""
A list of columns that is specific to the game, and not to the teams.
"""

def merge_game_data(df_games: pd.DataFrame, df_gameflow: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DataFrame in which each row contains the game data for a specific game for both teams. It receives a
    game flow DataFrame containing result changes for each game. It also receives a data frame with the
    game data where each row represents one team's view of the game.

    Params:
        df_games (pd.DataFrame): DataFrame with game data
        df_gameflow (pd.DataFrame): DataFrame with game flow data containing the result changes
        outcome_level (str): Level of the outcome to be modeled. Either 'game' or 'period'

    Returns:
        pd.DataFrame: DataFrame with game data merged for both teams
    """
    outcomes = (
        df_gameflow
            .sort_values(
                by=['game_id', 'period', 'period_time_remaining', 'home_score', 'away_score'],
                ascending=[True, True, False, True, True]
            )
            .groupby('game_id').tail(1)
    )
    outcomes['win'] = (outcomes['away_score'] > outcomes['home_score']).astype(int)
    df_games_filtered = df_games[df_games['game_id'].isin(outcomes['game_id'])].drop(columns=['season_id'])
    games_home_tmp = df_games_filtered[cols_base + cols_team].rename(columns=add_prefix(cols_team, 'home'))
    games_away_tmp = df_games_filtered[cols_base + cols_team].rename(columns=add_prefix(cols_team, 'away'))
    games_away_tmp['win'] = 1 - games_away_tmp['win']
    result = pd.merge(games_home_tmp, outcomes, left_on=['game_id', 'win'], right_on=['game_id', 'win'])
    result = pd.merge(result, games_away_tmp.drop(columns='game_date'), left_on=['game_id', 'win'], right_on=['game_id', 'win'])
    result = (
        result.drop(columns=['period', 'period_time_remaining'])
        .rename(columns={
            'home_score': 'home_score_final',
            'away_score': 'away_score_final',
            'win': 'away_win'
        })
    )
    return result
