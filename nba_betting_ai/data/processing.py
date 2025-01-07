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
    columns = ['season_id', 'game_id', 'game_date', 'result']
    teams = games_df['matchup'].str.split(expand=True)[[0, 2]].rename(columns={0: 'home_team', 2: 'away_team'})
    games_df = pd.concat([games_df[columns], teams], axis=1)
    return games_df