import numpy as np
import pandas as pd




def format_games_df(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the games DataFrame.
    
    Params:
        games_df (pd.DataFrame): DataFrame with games

    Returns:
        pd.DataFrame: Formatted DataFrame
    """
    games_df = games_df.copy()
    games_df['result'] = games_df['wl'].map({'W': 1, 'L': 0})
    columns = ['season_id', 'game_id', 'game_date', 'result']
    teams = games_df['matchup'].str.split(expand=True)[[0, 2]].rename(columns={0: 'home_team', 2: 'away_team'})
    games_df = pd.concat([games_df[columns], teams], axis=1)
    return games_df


def prepare_game_data(df_games: pd.DataFrame, df_gameflow: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the game data with results and team form.

    Params:
        df_games (pd.DataFrame): DataFrame with games
        df_gameflow (pd.DataFrame): DataFrame with game flow data

    Returns:
        pd.DataFrame: DataFrame with game data
    """
    df_results = df_gameflow[df_gameflow['game_id'].isin(df_gameflow['game_id'].unique())].groupby('game_id').tail(1).drop(columns=['time_remaining'])
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
    df_games[['season_wins','season_pts_for','season_pts_against_so_far']] = (
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
        
    return df_games

def add_prefix(to_list: list[str], prefix: str) -> dict[str, str]:
    """
    Add a prefix to each element in the list.

    Params:
        to_list (list[str]): List of elements
        prefix (str): Prefix

    Returns:
        dict[str, str]: Dictionary with the elements and the prefix
    """
    return {
        elem: f'{prefix}_{elem}'
        for elem in to_list
    }

cols_team = [
    'team_id',
    'team_abbreviation',
    'season_wins',
    'season_pts_for',
    'season_pts_against_so_far',
    'season_games',
    'last_5_wins',
    'last_5_pts_for_avg',
    'last_5_pts_for_total',
    'last_5_pts_against_avg',
    'last_5_pts_against_total'
]
"""
A list of columns related to each teams statistics.
"""
cols_base = ['game_id', 'win']
"""
A list of columns that is specific to the game, and not to the teams.
"""

def merge_data_for_indices(index: list[int], df_games: pd.DataFrame, df_gameflow: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame with the game data for the given indices. Dataframe consists of home and away team statistics,
    together with the game result, and game and team identifiers.

    Params:
        index (list[int]): List of indices
        df_games (pd.DataFrame): DataFrame with games
        df_gameflow (pd.DataFrame): DataFrame with game flow data

    Returns:
        pd.DataFrame: DataFrame with the game data
    """
    game_ids = df_gameflow.loc[index]['game_id']
    outcomes = df_gameflow[df_gameflow['game_id'].isin(game_ids)].groupby('game_id').tail(1)
    outcomes['win'] = (outcomes['home_score'] > outcomes['away_score']).astype(int)
    df_games_filtered = df_games[df_games['game_id'].isin(outcomes['game_id'])].drop(columns=['season_id'])
    games_home_tmp = df_games_filtered[cols_base + cols_team].rename(columns=add_prefix(cols_team, 'home'))
    games_away_tmp = df_games_filtered[cols_base + cols_team].rename(columns=add_prefix(cols_team, 'away'))
    games_away_tmp['win'] = 1 - games_away_tmp['win']
    result = pd.merge(games_home_tmp, outcomes, left_on=['game_id', 'win'], right_on=['game_id', 'win'])
    result = pd.merge(result, games_away_tmp, left_on=['game_id', 'win'], right_on=['game_id', 'win'])
    result.drop(columns=['home_score', 'away_score', 'time_remaining'], inplace=True)
    return result
