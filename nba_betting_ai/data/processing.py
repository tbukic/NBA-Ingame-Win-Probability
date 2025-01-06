import pandas as pd


def format_games_df(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the games DataFrame.
    
    Params:
        games_df (pd.DataFrame): DataFrame with games

    Returns:
        pd.DataFrame: Formatted DataFrame
    """
    games_df['result'] = games_df['wl'].map({'w': 1, 'l': 0})
    columns = ['SEASON_ID', 'GAME_ID', 'GAME_DATE', 'RESULT']
    teams = games_df['MATCHUP'].str.split(expand=True)[[0, 2]].rename(columns={0: 'HOME_TEAM', 2: 'AWAY_TEAM'})
    games_df = pd.concat([games_df[columns], teams], axis=1)
    return games_df