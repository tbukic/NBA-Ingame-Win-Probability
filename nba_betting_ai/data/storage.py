import pandas as pd
import subprocess

from datetime import datetime
from pathlib import Path
from sqlalchemy import Engine, inspect, text

from nba_betting_ai.consts import proj_paths


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

def export_postgres_db(
    db_name: str,
    output_file: Path = None,
    host: str = "localhost",
    port: str = "5432",
    username: str = "postgres",
    schema: str = None,
    password: str = None
) -> Path:
    """
    Export a PostgreSQL database to a file.

    Params:
        db_name (str): Database name
        output_file (Path): Output file path
        host (str): Host name
        port (str): Port number
        username (str): Username
        schema (str): Schema name
        password (str): Password

    Returns:
        Path: Path to the output file
    """
    if not output_file:
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = proj_paths.pg_dump / f"{db_name}_{date_str}.sql"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'pg_dump',
        '-h', host,
        '-p', port,
        '-U', username,
        '-F', 'c',
        '-b',
        '-v',
        '-f', output_file.as_posix(),
    ]
    
    # Add schema if specified
    if schema:
        cmd.extend(['-n', schema])
    
    cmd.append(db_name)
    
    # Set PostgreSQL password environment variable
    env = {'PGPASSWORD': password}
    
    subprocess.run(cmd, env=env, check=True)
    print(f"Database successfully exported to {output_file}")
    return output_file

def database_empty(engine: Engine,) -> bool:
    """
    Check if a database is empty.

    Params:
        engine (Engine): SQLAlchemy engine

    Returns:
        bool: True if the database is empty, False otherwise
    """
    with engine.connect() as conn:
        query = text("""
            SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public'
        )
        """)
        result = conn.execute(query)
        is_empty = not result.scalar()
    return is_empty

def import_postgres_db(
    engine: Engine,
    backup_file: Path,
    db_name: str,
    host: str = "localhost",
    port: str = "5432",
    username: str = "postgres",
    password: str = None,
    nonempty_proceed: bool = False
) -> None:
    """
    Import a PostgreSQL database from a file.

    Params:
        engine (Engine): SQLAlchemy engine
        backup_file (Path): Backup file path
        db_name (str): Database name
        host (str): Host name
        port (str): Port number
        username (str): Username
        password (str): Password
        nonempty_proceed (bool): If True, proceed even if the database is not empty

    Raises:
        Exception: If the database already exists and is not empty
    """
    db_empty = database_empty(engine)
    if not db_empty and not nonempty_proceed:
        raise Exception(f"Database {db_name} already exists and is not empty")
    
    restore_cmd = [
        'pg_restore',
        '-h', host,
        '-p', port,
        '-U', username,
        '-d', db_name,
        '-v',
        '--clean',
        '--if-exists',
        '-1',
        backup_file.as_posix(),
    ]
    
    env = {'PGPASSWORD': password}
    subprocess.run(restore_cmd, env=env, check=True)
    print(f"Database successfully restored from {backup_file}")
