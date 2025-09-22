import logging
import os
import pandas as pd
import seaborn as sns
import streamlit as st
import threading
import time
from attrs import frozen
from bokeh.plotting import figure
from catboost import CatBoostRegressor
from collections import deque
from datetime import datetime
from functools import partial
from itertools import cycle
from nba_api.stats.library.parameters import SeasonAll
from pathlib import Path
from torch import nn
from typing import Any

from nba_betting_ai.consts import proj_paths
from nba_betting_ai.data.ingest import scrape_everything
from nba_betting_ai.data.storage import check_table_exists, database_empty, get_engine, import_postgres_db
from nba_betting_ai.deploy.inference_bayesian import calculate_probs_bayesian, load_bayesian_model, prepare_experiment
from nba_betting_ai.deploy.inference_catboost import load_catboost_model, calculate_probs_catboost
from nba_betting_ai.deploy.inference_linear import load_linear_model, calculate_probs_linear
from nba_betting_ai.deploy.utils import Line
from nba_betting_ai.model.inputs import Scalers
from nba_betting_ai.training.pipeline import prepare_data


def check_database_exists() -> bool:
    engine = get_engine()
    if database_empty(engine):
        return False
    if not all(
        check_table_exists(engine, table)
        for table in _tables
    ):
        return False
    return True

@st.cache_resource()
def pg_get_engine() -> Any:
    return get_engine()

engine = pg_get_engine()

_tables= ('games', 'teams', 'gameflow')

def upload_database(file: Path):
    postgres_user = os.environ.get('POSTGRES_USER')
    postgres_password = os.environ.get('POSTGRES_PASSWORD')
    postgres_host = os.environ.get('POSTGRES_HOST')
    postgres_port = os.environ.get('POSTGRES_PORT')
    postgres_db = os.environ.get('POSTGRES_DB')
    now = time.strftime("%Y%m%d-%H%M%S")
    backup_file = proj_paths.pg_dump / f'upload_{now}.sql'
    file_content = file.read()
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    backup_file.write_bytes(file_content)
    import_postgres_db(
        engine=get_engine(),
        backup_file=backup_file,
        db_name=postgres_db,
        username=postgres_user,
        password=postgres_password,
        host=postgres_host,
        port=postgres_port,
        nonempty_proceed=True
    )

@st.dialog("Upload database")
def prompt_upload_database():
    # st.header("Importing the database")
    uploaded_file = st.file_uploader("Choose a file", type=['sql'])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        upload_database(uploaded_file)
        st.session_state['database_ready'] = True
        st.rerun()
    elif not st.session_state.get('database_ready', False):
        st.stop()

color_cycle = cycle(sns.color_palette("husl"))
game_limit = 10

@st.cache_data
def get_paths(*paths, prefix: str) -> tuple[Path]:
    return tuple(
        proj_paths.models / Path(path).with_stem(f'{prefix}_{Path(path).stem}')
        for path in paths
    )

@frozen
class ModelStore:
    model: nn.Module | CatBoostRegressor
    scalers: Scalers
    team_features: list
    config: Any

@frozen
class DataStore:
    X: pd.DataFrame
    teams: pd.DataFrame
    team_encoder: dict[str, int]
    store: ModelStore

@st.cache_resource()
def load_resources(model_path: Path, config_path: Path, scalers_path: Path) -> DataStore:
    # Determine model type based on file extension
    if model_path.suffix == '.pth':
        # PyTorch model (Bayesian)
        model_init_path = config_path.with_suffix('.yaml').with_stem(config_path.stem.replace('run_config', 'model_init'))
        model, scalers, team_features, config = load_bayesian_model(model_path, model_init_path, config_path, scalers_path)
    elif model_path.suffix == '.cbm':
        # CatBoost model
        model, scalers, team_features, config = load_catboost_model(model_path, config_path, scalers_path)
    elif model_path.suffix == '.pkl':
        # Linear/Logistic model
        model, scalers, team_features, config = load_linear_model(model_path, config_path, scalers_path)
    else:
        raise ValueError(f"Unsupported model file type: {model_path.suffix}")
    
    scalers.pop('final_score_diff', None)
    data_params = {
        'seasons': 1,
        'seed': 0,
        'test_size': 1.0,
        'n': None,
        'frac': 1.0,
    }
    data_prepared = prepare_data(**data_params)
    X = data_prepared.X_test
    teams = data_prepared.teams
    team_encoder = dict(zip(teams['abbreviation'], teams.index))
    store = ModelStore(model, scalers, team_features, config)
    return DataStore(
        X, teams, team_encoder, store
    )

def init_team_features(side: str, team_features: list[str]):
    team_abbrev = st.session_state[f'{side}_team']
    dummy_experiment = get_matchup_data(team_abbrev, team_abbrev)
    start_data = dummy_experiment.iloc[0]
    for feature in team_features:
        st.session_state[feature] = start_data[feature]

@st.fragment()
def select_team(side: str):
    data_store = st.session_state.data_store
    team_features = [
        feature
        for feature in data_store.store.team_features
        if side in feature
    ]
    st.selectbox(
        f"{side.title()} Team",
        key=f'{side}_team',
        options=data_store.teams['abbreviation'].tolist(),
        placeholder=f"Pick a {side} team",
        on_change=partial(init_team_features, side, team_features)
    )
    if team_features and team_features[0] not in st.session_state:
        init_team_features(side, team_features)
    for feature in team_features:
        step = 0.2 if 'avg' in feature else 1
        st.number_input(
            feature,
            key=feature,
            step=step,
            format="%.2f" if 'avg' in feature else "%d"
        )

if 'matchups' not in st.session_state:
    st.session_state.matchups = {}
if 'plot_colors' not in st.session_state:
    st.session_state.plot_colors = {}
if 'plot_data' not in st.session_state:
    st.session_state.plot_data = {}
if 'matchup_models' not in st.session_state:
    st.session_state.matchup_models = {}
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}

@st.cache_data
def get_matchup_data(home_abbrev, away_abbrev):
    data_store = st.session_state.data_store
    score_diff = st.session_state.get('score_diff', 0)
    experiment = prepare_experiment(home_abbrev, away_abbrev, score_diff, data_store.X.copy(), data_store.teams)
    return experiment

def generate_matchup_name(home_abbrev, away_abbrev) -> str:
    model_name = st.session_state.model
    simple_name = f'{away_abbrev} @ {home_abbrev} ({model_name})'
    if simple_name not in st.session_state['matchups']:
        return simple_name
    i = 1
    while (name:=f'{simple_name} ({i})') in st.session_state['matchups']:
        i += 1
    return name

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def get_color():
    while (color:=rgb_to_hex(next(color_cycle))) in st.session_state.plot_colors.values():
        pass
    return color

def load_and_cache_model(model_name: str) -> DataStore:
    """Load a model and cache it for reuse. Returns the cached model if already loaded."""
    if model_name in st.session_state.model_cache:
        return st.session_state.model_cache[model_name]
    
    try:
        model_path = _find_model_path(model_name)
        config_path = model_path.parent / 'run_config.yaml'
        scalers_path = model_path.parent / 'scalers.pkl'
        
        # Load the model resources
        model_data_store = load_resources(model_path, config_path, scalers_path)
        
        # Cache the loaded model
        st.session_state.model_cache[model_name] = model_data_store
        
        return model_data_store
        
    except Exception as e:
        st.error(f"Failed to load model '{model_name}': {str(e)}")
        raise e

@st.cache_data
def get_plot_data(home_abbrev, away_abbrev, experiment: pd.DataFrame, score_diff: float, model_name: str) -> Line:
    experiment = experiment.copy()
    experiment['score_diff'] = score_diff
    
    # Load the correct model resources for this specific model (using cache)
    try:
        model_data_store = load_and_cache_model(model_name)
        model_path = _find_model_path(model_name)
        
        # Determine the correct probability function based on model type
        if model_path.suffix == '.pth':
            prob_function = calculate_probs_bayesian
        elif model_path.suffix == '.cbm':
            prob_function = calculate_probs_catboost
        elif model_path.suffix == '.pkl':
            prob_function = calculate_probs_linear
        else:
            st.error(f"Unsupported model type: {model_path.suffix}")
            return Line(home_team=home_abbrev, away_team=away_abbrev, data=pd.DataFrame(), score_diff=score_diff)
        
    except Exception as e:
        st.error(f"Failed to load model '{model_name}': {str(e)}")
        return Line(home_team=home_abbrev, away_team=away_abbrev, data=pd.DataFrame(), score_diff=score_diff)
    
    results = prob_function(
        abbrev_home=home_abbrev,
        abbrev_away=away_abbrev,
        experiment=experiment,
        model=model_data_store.store.model,
        config=model_data_store.store.config,
        scalers=model_data_store.store.scalers,
        team_features=model_data_store.store.team_features, 
        team_encoder=model_data_store.team_encoder
    )
    # Denormalizing time remaining
    results.loc[:, 'game_time'] = experiment['time_remaining'].max() - experiment['time_remaining']
    line = Line(home_abbrev, away_abbrev, results[['game_time', 'probs']], score_diff)
    return line

@st.dialog("Pick matchup name")
def add_named_matchup(experiment: pd.DataFrame):
    home_abbrev = st.session_state.home_team
    away_abbrev = st.session_state.away_team
    default_name = generate_matchup_name(
        home_abbrev=home_abbrev,
        away_abbrev=away_abbrev
    )
    name = st.text_input("Matchup name", value=default_name)
    if st.button("Save"):
        if len(st.session_state.matchups) >= game_limit:
            st.error(f"Cannot add more than {game_limit} matchups. Delete some to add more.")
            return
        st.session_state.matchups[name] = experiment
        st.session_state.plot_colors[name] = get_color()
        st.session_state.matchup_models[name] = st.session_state.model
        st.session_state.plot_data[name] = get_plot_data(home_abbrev, away_abbrev, experiment, st.session_state.score_diff, st.session_state.model)
        st.rerun()

def add_matchup():
    for team in ('home', 'away'):
        if st.session_state.get(f'{team}_team') is None:
            st.error(f"Please select {team} team")
            return
    experiment = get_matchup_data(st.session_state.home_team, st.session_state.away_team)
    data_store = st.session_state.data_store
    for feature in data_store.store.team_features:
        experiment[feature] = st.session_state[feature]
    add_named_matchup(experiment)

def select_teams():
    with st.container(border=True):
        select_team('away')
        st.markdown('---')
        select_team('home')
        st.button("Add matchup", on_click=add_matchup)

def draw_plots():
    if not st.session_state.plot_data:
        return
    p = figure(
        title="Win Probability",
        x_axis_label='Game Time [s]',
        y_axis_label='P(Home Win)',
        background_fill_color="white",
        width=600,
        height=400
    )
    for name, line in st.session_state.plot_data.items():
        p.line(
            line.data['game_time'],
            line.data['probs'], 
            line_width=2,
            legend_label=name,
            color=st.session_state.plot_colors[name]
        )
    p.legend.location = "top_left"
    p.x_range.start = 0
    p.x_range.end = 3000
    p.y_range.start = 0
    p.y_range.end = 1
    st.bokeh_chart(p, use_container_width=True)

logs = deque()

class StreamlitLogHandler(logging.Handler):
    def emit(self, record):
        logs.append(self.format(record))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = StreamlitLogHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

@st.dialog("Scrape data")
def scrape_new_data():
    kwargs = {
        'engine': engine,
    }
    if check_table_exists(engine, 'games'):
        last_date = pd.read_sql('SELECT game_date FROM games ORDER BY game_date DESC LIMIT 1', engine)
        last_date = datetime.strptime(last_date.iloc[0]['game_date'], '%Y-%m-%d').strftime('%m/%d/%Y') if not last_date.empty else None
        kwargs['start_date'] = last_date
        ingest_message = f'Scraping data from {last_date} to yesterday'
    else:
        kwargs['season'] = SeasonAll.current_season
        ingest_message = f'Scraping data for the current season ({kwargs["season"]})'
    thread = threading.Thread(target=scrape_everything, kwargs=kwargs)
    thread.start()
    with st.status(ingest_message):
        while thread.is_alive():
            while logs:
                st.write(logs.popleft())
            time.sleep(0.5)
        st.write("Process completed!")
    st.stop()

if not st.session_state.get('database_ready', check_database_exists()):
    with st.sidebar:
        st.button("Upload database", on_click=prompt_upload_database)
        st.button("Scrape data", on_click=scrape_new_data)
        st.rerun()

def delete_selected_games():
    for game in st.session_state.delete_select:
        st.session_state.matchups.pop(game)
        st.session_state.plot_colors.pop(game)
        st.session_state.plot_data.pop(game)
        st.session_state.matchup_models.pop(game, None)  # Use .pop(game, None) to avoid KeyError if key doesn't exist
    st.rerun()

def reparametrize_plots():
    """Regenerate all plots with the new score difference, using each plot's original model."""
    new_plot_data = {}
    
    for matchup_name in st.session_state.matchups.keys():
        # Get the model that was used for this matchup
        matchup_model = st.session_state.matchup_models.get(matchup_name, st.session_state.model)
        
        # Get the original plot data to extract team names
        if matchup_name in st.session_state.plot_data:
            original_line = st.session_state.plot_data[matchup_name]
            home_abbrev = original_line.home_team
            away_abbrev = original_line.away_team
            
            # Regenerate the plot data with the correct model
            new_plot_data[matchup_name] = get_plot_data(
                home_abbrev, 
                away_abbrev, 
                st.session_state.matchups[matchup_name], 
                st.session_state.score_diff, 
                matchup_model
            )
    
    st.session_state.plot_data = new_plot_data

def _collect_models():
    roots = [proj_paths.models, proj_paths.models / 'production']
    found = []
    for root in roots:
        if root.exists():
            for p in root.rglob('*'):
                if p.suffix in ('.pth', '.cbm', '.pkl') and p.stem == 'model':
                    model_type = p.parent.name
                    found.append(model_type)
    return sorted(set(found), reverse=True)

def _find_model_path(name: str) -> Path:
    roots = [proj_paths.models, proj_paths.models / 'production']
    for root in roots:
        if not root.exists():
            continue
        model_dir = root / name
        if model_dir.exists():
            for p in model_dir.iterdir():
                if p.suffix in ('.pth', '.cbm', '.pkl') and p.stem == 'model':
                    return p
    raise FileNotFoundError(name)

def preload_all_models():
    """Preload all available models to cache them for faster access."""
    model_options = _collect_models()
    with st.status("Loading models...") as status:
        for i, model_name in enumerate(model_options):
            try:
                status.update(label=f"Loading {model_name}...")
                load_and_cache_model(model_name)
            except Exception as e:
                st.warning(f"Failed to preload model '{model_name}': {str(e)}")
        status.update(label="All models loaded!", state="complete")

def select_model():
    model_name = st.session_state.model
    try:
        # Use the cached model loading
        st.session_state.data_store = load_and_cache_model(model_name)
        
        model_path = _find_model_path(model_name)
        if model_path.suffix == '.pth':
            st.session_state.prob_function = calculate_probs_bayesian
        elif model_path.suffix == '.cbm':
            st.session_state.prob_function = calculate_probs_catboost
        elif model_path.suffix == '.pkl':
            st.session_state.prob_function = calculate_probs_linear
            
        # Don't regenerate existing plots when changing model selection
        # Only the newly added matchups should use the new model
        
    except Exception as e:
        st.error(f"❌ Failed to load model '{model_name}': {str(e)}")
        # Keep previous model if loading fails
        if 'data_store' not in st.session_state:
            st.error("No fallback model available!")
        else:
            st.warning("Keeping previous model loaded")

model_options = _collect_models()

# Preload all models if not already done
if 'models_preloaded' not in st.session_state:
    preload_all_models()
    st.session_state.models_preloaded = True

with st.sidebar:
    st.header("Select model")
    st.selectbox(
        "Model",
        key='model',
        options=model_options,
        on_change=select_model
    )
    if 'data_store' not in st.session_state:
        select_model()
    st.header("Add New Matchup")
    select_teams()
    score_diff = st.slider("Score Difference: [Away - Home]", -50, 50, 0, step=1, key='score_diff', on_change=reparametrize_plots)
    st.markdown('---')
    select_to_delete = st.multiselect(
        options=list(st.session_state.matchups.keys()),
        key="delete_select",
        label="Select games to delete",
    )
    if select_to_delete:
        st.button("Delete selected games", on_click=delete_selected_games)
    st.markdown('---')
    st.header("Upgrade database")
    col_left, col_right = st.columns(2)
    with col_left:
        st.button("Scrape data", on_click=scrape_new_data)
    with col_right:
        st.button("New database", on_click=prompt_upload_database)
    
draw_plots()
