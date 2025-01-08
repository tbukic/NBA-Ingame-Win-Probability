import logging
import pandas as pd
import random

from datetime import datetime
from nba_api.stats.endpoints import playbyplayv2, leaguegamefinder
from nba_api.stats.static.teams import get_teams
from sqlalchemy import Engine
from tenacity import retry, wait_random_exponential, stop_after_attempt, before_log, retry_if_not_exception_type
from time import sleep

from nba_betting_ai.data.storage import check_table_exists, load_teams, load_games, get_uningested_games

logger = logging.getLogger(__name__)

games_cols = ['season_id', 'team_id', 'team_abbreviation', 'team_name', 'game_id', 'game_date', 'matchup', 'wl']
gameflow_cols = ['game_id', 'home_score', 'away_score', 'time_remaining']

_wait=wait_random_exponential(multiplier=1, max=60)
_stop=stop_after_attempt(15)
_before = before_log(logger=logger, log_level=logging.DEBUG)


@retry(wait=_wait, stop=_stop, before=_before)
def scrape_teams(headers: dict | None = None) -> pd.DataFrame:
    """
    Scrape teams from the NBA API.

    Returns:
        pd.DataFrame: DataFrame with teams
    """
    teams = get_teams(headers=headers)
    teams_df = pd.DataFrame(teams)
    return teams_df


def check_date_format(date: str | None) -> None:
    """
    Check if the date is in the correct format MM/DD/YYYY

    Param:
        date (str): Date in MM/DD/YYYY format

    Raises:
        ValueError: If the date is not in the correct format
    """
    if not date:
        return
    try:
        datetime.strptime(date, '%m/%d/%Y')
    except ValueError:
        raise ValueError("Incorrect date format, should be MM/DD/YYYY")


def format_season_id(season_id: str) -> str:
    """
    Format the season ID to the correct format YYYY-YY

    Param:
        season_id (str): Season ID in the format iYYYY

    Returns:
        str: Season ID in the format YYYY-YY
    """
    year = str(season_id)[-4:]
    next_year = int(year) + 1
    return f"{year}-{str(next_year)[-2:]}"


_yesterday = datetime.now().strftime('%m/%d/%Y')

@retry(wait=_wait, stop=_stop, before=_before, retry=retry_if_not_exception_type(ValueError))
def scrape_games_between(season: str | None = None, start_date: str | None = None, end_date: str | None = None, timeout: int = 60, headers: dict | None = None) -> pd.DataFrame:
    """
    Scrape games between two dates from the NBA API. If end_date is today or later put it to yesterday so
    no unfinished games are included.

    Params:
        season (str): Season to scrape
        start_date (str): Start date in MM/DD/YYYY format
        end_date (str): End date in MM/DD/YYYY format. Latest possible date is yesterday.
        timeout (int): Timeout for the API request

    Returns:
        pd.DataFrame: DataFrame with games between the two dates
    """
    if all([season is None, start_date is None, end_date is None]):
        raise ValueError("At least one of season, start_date or end_date must be provided.")
    check_date_format(start_date)
    check_date_format(end_date)
    if end_date and end_date > _yesterday:
        end_date = _yesterday
    gamefinder = leaguegamefinder.LeagueGameFinder(
        date_from_nullable=start_date,
        date_to_nullable=end_date,
        season_nullable=season,
        team_id_nullable=None,
        league_id_nullable='00',  # NBA games only
        timeout=timeout,
        headers=headers
    )
    games_dict = gamefinder.get_normalized_dict()
    games_df = pd.DataFrame(games_dict['LeagueGameFinderResults'])
    games_df.columns = games_df.columns.str.lower()
    games_df['season_id'] = games_df['season_id'].apply(format_season_id)
    #raw_cols = ['SEASON_ID', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL']
    #games_df = games_df[raw_cols]
    return games_df


# @retry(wait=_wait, stop=_stop, before=_before)
def scrape_gameflow(game_id: str, timeout: int = 60, headers: str | None = None) -> pd.DataFrame:
    """
    Scrape the game flow for a game from the NBA API.

    Params:
        game_id (str): Game ID
        timeout (int): Timeout for the API request

    Returns:
        pd.DataFrame: DataFrame with the game flow for the game
    """
    play_by_play = playbyplayv2.PlayByPlayV2(game_id=game_id, timeout=timeout, headers=headers)
    try:
        df_plays = play_by_play.get_data_frames()[0]
    except Exception as e:
        logger.error(f"Error scraping gameflow for game {game_id}: {e}")
        raise e
    df_plays.columns = df_plays.columns.str.lower()
    scored_mask = ~df_plays['score'].isna()
    df_plays = df_plays[scored_mask]
    if df_plays.empty:
        return df_plays
    scores = df_plays['score'].str.split(' - ', expand=True)
    scores.columns = ['home_score', 'away_score']
    scores = scores.astype(int)
    scores.insert(0, 'game_id', game_id)
    # scores['DIFF'] = scores['HOME_SCORE'] - scores['AWAY_SCORE']
    period_length = 12 * 60
    total_periods = df_plays['period'].max()
    time_remaining_period = df_plays['pctimestring'].str.split(':', expand=True).astype(int)
    time_remaining_period = time_remaining_period[0] * 60 + time_remaining_period[1]
    time_remaining = (total_periods - df_plays['period']) * period_length + time_remaining_period
    scores['time_remaining'] = time_remaining
    return scores


def ingest_games(engine: Engine, season: str | None, start_date: str | None, end_date: str | None, headers: dict|str = None) -> pd.Series:
    """
    Ingest games from the NBA API.

    Params:
        engine (Engine): SQLAlchemy engine
        season (str): Season to scrape
        start_date (str): Start date in MM/DD/YYYY format
        end_date (str): End date in MM/DD/YYYY format
        headers (dict): Headers for the API request

    Returns:
        pd.Series: Series with the game IDs
    """
    table_exists = check_table_exists(engine, 'games')
    games_existing = load_games(engine, season_id=season, just_ids=True) if table_exists else None
    games_df = scrape_games_between(season, start_date, end_date, headers=headers)
    games_too_recent = games_df[games_df['game_date'] >= _yesterday]
    games_unfinished = games_too_recent[games_too_recent['wl'].isna()]
    if not games_unfinished.empty:
        logger.info(f'Found {len(games_unfinished)} unfinished games from today. Removing them.')
        games_df = games_df[~games_df['game_id'].isin(games_unfinished['game_id'])]
    if games_existing is not None:
        games_df = games_df[~games_df['game_id'].isin(games_existing['game_id'])]
    games_df.columns = games_df.columns.str.lower()
    games_df = games_df[games_cols]
    if games_df.empty:
        return pd.Series([])
    games_df.to_sql('games', engine, if_exists='append', index=False)
    logger.info(f'Ingested {len(games_df)} new games.')
    return games_df['game_id']


def ingest_teams(engine: Engine = None, headers: dict|None=None, force_scrape: bool = False) -> pd.Series:
    """
    Ingest teams from the NBA API.

    Params:
        engine (Engine): SQLAlchemy engine
        headers (dict): Headers for the API request
        force_scrape (bool): If True, scrape teams even if they already exist in the database

    Returns:
        pd.Series: Series with the team IDs
    """
    table_exists = check_table_exists(engine, 'teams')
    teams_existing = load_teams(engine, just_ids=True) if table_exists else None
    if teams_existing is not None and not force_scrape:
        return teams_existing['team_id']
    teams_df = scrape_teams(headers=headers)
    if teams_existing is not None:
        teams_df = teams_df[~teams_df['id'].isin(teams_existing['team_id'])]
    teams_df.columns = teams_df.columns.str.lower()
    teams_df = teams_df.rename(columns={'id': 'team_id'})
    if teams_df.empty:
        return pd.Series([])
    teams_df.to_sql('teams', engine, if_exists='append', index=False)
    logger.info(f'Ingested {len(teams_df)} new teams.')
    return teams_df['team_id']

def ingest_gameflow(engine: Engine, game_id: str, headers: str | None = None) -> None:
    """
    Ingest game flow from the NBA API.

    Params:
        engine (Engine): SQLAlchemy engine
        game_id (str): Game ID
        headers (dict): Headers for the API request
    """
    gameflow_df = scrape_gameflow(game_id, headers=headers)
    if gameflow_df.empty:
        logger.info(f'No gameflow found for game {game_id}.')
        return
    gameflow_df.columns = gameflow_df.columns.str.lower()
    gameflow_df = gameflow_df[gameflow_cols]
    gameflow_df.to_sql('gameflow', engine, if_exists='append', index=False)
    logger.info(f'Ingested gameflow for game {game_id}.')


def wait_random(min_wait: float=0.6, max_wait: float=1.2) -> None:
    random_time = min_wait + (max_wait - min_wait)*random.random()
    logger.info(f'Sleeping for {random_time} sec.')
    sleep(random_time)


def scrape_everything(
        engine: Engine,
        season: str | None = None,
        start_date: str | None = '10/22/2023',
        end_date: str | None = _yesterday,
        headers: dict | None = None
    ) -> None:
    """
    Scrape teams, games and game flows from the NBA API.

    Params:
        engine (Engine): SQLAlchemy engine
        season (str): Season to scrape
        start_date (str): Start date in MM/DD/YYYY format
        end_date (str): End date in MM/DD/YYYY format. Latest possible date is yesterday.
    """
    logger.info('Ingesting teams.')
    ingest_teams(engine, headers=headers)
    load_teams(engine, just_ids=True)
    wait_random()
    logger.info('Ingesting games.')
    ingest_games(engine, season, start_date, end_date, headers=headers)
    id_games_new = get_uningested_games(engine)
    logger.info('Ingesting new gameflows.')
    for pos, game in enumerate(id_games_new):
        wait_random()
        logger.info(f'Ingesting gameflow for game {game} ({pos + 1}/{len(id_games_new)}).')
        ingest_gameflow(engine, game, headers=headers)
    logger.info('Done ingesting games and gameflows.')

