import pandas as pd
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError


def check_table_exists(engine: Engine, table_name: str) -> bool:
    """
    Check if a table exists in the database.

    Params:
        engine (Engine): SQLAlchemy engine
        table_name (str): Table name to check

    Returns:
        bool: True if the table exists, False otherwise
    """
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def load_teams(engine: Engine, just_ids: bool = False) -> pd.DataFrame | None:
    """
    Load teams from the database.

    Params:
        engine (Engine): SQLAlchemy engine
        just_ids (bool): If True, return only the IDs

    Returns:
        pd.DataFrame | None: DataFrame with teams or None if no teams are found
    
    Raises:
        SQLAlchemyError: If a database error occurs
        Exception: If an unexpected error occurs
    """
    querry = text("SELECT * FROM teams") if not just_ids else text("SELECT team_id FROM teams")
    df_teams = pd.read_sql(querry, engine)
    return df_teams if not df_teams.empty else None


def load_games(engine: Engine, season_id: str | None = None, just_ids: bool = False) -> pd.DataFrame | None:
    """
    Load games for a season from the database.

    Params:
        engine (Engine): SQLAlchemy engine
        season_id (str): Season ID to filter games
        just_ids (bool): If True, return only the IDs

    Returns:
        pd.DataFrame | None: DataFrame with games or None if no games are found

    Raises:
        SQLAlchemyError: If a database error occurs
        Exception: If an unexpected error occurs
    """
    query_text = "SELECT * FROM games" if not just_ids else "SELECT game_id FROM games"
    if season_id:
        query_text += " WHERE season_id = :season_id"
        params = {"season_id": season_id}
    else:
        params = {}
    
    df_games = pd.read_sql(text(query_text), engine, params=params)
    return df_games if not df_games.empty else None


def load_gameflow(engine: Engine, game_id: str | list[str], only_ids: bool=False) -> pd.DataFrame | None:
    """
    Load the game flow for a game from the database.

    Params:
        engine (Engine): SQLAlchemy engine
        game_id (str): Game ID to filter game flow

    Returns:
        pd.DataFrame | None: DataFrame with game flow or None if no game flow is found

    Raises:
        SQLAlchemyError: If a database error occurs
        Exception: If an unexpected error occurs
    """
    query_text = "SELECT * FROM gameflow" if not only_ids else "SELECT game_id FROM gameflow"
    if isinstance(game_id, str):
        query_condition = " WHERE game_id = :game_id"
        params = {"game_id": game_id}
    elif isinstance(game_id, list):
        query_condition = " WHERE game_id IN :game_id"
        params = {"game_id": tuple(game_id)}
    else:
        raise ValueError(f"Unsupported game_id type: {type(game_id)}")
    query_text += query_condition
    df_gameflow = pd.read_sql(text(query_text), engine, params=params)
    return df_gameflow if not df_gameflow.empty else None

def get_uningested_games(engine: Engine) -> pd.Series:
    """
    Get the uningested games from the NBA API.

    Params:
        engine (Engine): SQLAlchemy engine
    
    Returns:
        pd.Series: Series with the game IDs
    """
    games_existing = load_games(engine, just_ids=True)
    if games_existing is None or games_existing.empty:
        return pd.Series()
    
    gameflow_exists = check_table_exists(engine, 'gameflow')
    if not gameflow_exists:
        return games_existing['game_id']
    
    processed_games = load_gameflow(engine, game_id=games_existing['game_id'].tolist(), only_ids=True)
    if processed_games is None or processed_games.empty:
        return games_existing['game_id']
    
    existing_set = set(games_existing['game_id'])
    processed_set = set(processed_games['game_id'])
    
    unprocessed_games = pd.Series(list(existing_set - processed_set))
    return unprocessed_games
