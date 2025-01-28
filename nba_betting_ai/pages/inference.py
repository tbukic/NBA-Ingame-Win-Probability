import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import streamlit as st
import time
from attrs import frozen
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from itertools import cycle

from pathlib import Path
from torch import nn
from typing import Any

from nba_betting_ai.consts import proj_paths
from nba_betting_ai.data.storage import check_table_exists, database_empty, get_engine, import_postgres_db
from nba_betting_ai.deploy.inference import calculate_probs_for_diff, load_model, prepare_experiment
from nba_betting_ai.model.inputs import Scalers
from nba_betting_ai.training.pipeline import prepare_data


# status_placeholder = st.empty()

_postgres_user = os.environ.get('POSTGRES_USER')
_postgres_password = os.environ.get('POSTGRES_PASSWORD')
_postgres_host = os.environ.get('POSTGRES_HOST')
_postgres_port = os.environ.get('POSTGRES_PORT')
_postgres_db = os.environ.get('POSTGRES_DB')
_tables= ('games', 'teams', 'gameflow')


@st.cache_resource()
def check_database_exists(newest: tuple[Path, str]) -> bool:
    engine = get_engine()
    if database_empty(engine):
        return False
    if not all(
        check_table_exists(engine, table)
        for table in _tables
    ):
        return False
    return True

def upload_database(file: Path):
    now = time.strftime("%Y%m%d-%H%M%S")
    backup_file = proj_paths.pg_dump / f'{now}.sql'
    file_content = file.read()
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    backup_file.write_bytes(file_content)
    import_postgres_db(
        engine=get_engine(),
        backup_file=backup_file,
        db_name=_postgres_db,
        username=_postgres_user,
        password=_postgres_password,
        host=_postgres_host,
        port=_postgres_port,
        nonempty_proceed=True
    )

def find_newest() -> tuple[Path, str]:
    if not proj_paths.pg_dump.exists():
        return proj_paths.pg_dump, "None"
    newest = max(proj_paths.pg_dump.glob('*.sql'), key=lambda f: f.stat().st_mtime)
    if newest is None:
        return proj_paths.pg_dump, "None"
    time = str(newest.stat().st_mtime)
    return newest, time

# This is used only for caching.
newest_flag = find_newest()
database_exists = check_database_exists(newest_flag)
if not database_exists:
    with st.sidebar:
        st.header("Importing the database")
        #if st.button("Import Database"):
        uploaded_file = st.file_uploader("Choose a file", type=['sql'])
        if uploaded_file is not None:
            st.write("File uploaded successfully!")
            upload_database(uploaded_file)
        else:
            st.warning("Please upload a database file first!")
            st.stop()

color_cycle = cycle(sns.color_palette("husl"))
game_limit = 10

wpaths_names = (
    'run_config.yaml', 'scalers.pkl',
    'bayesian_model-20250128-120945-loss_1_2943673074375524.pth',
    'model_init.yaml'
)
nwpaths_names = (
    'run_config.yaml', 'scalers.pkl',
    'bayesian_model-20250128-125400-loss_1_3243823564165846.pth',
    'model_init.yaml'
)

def get_paths(*paths, prefix: str) -> tuple[Path]:
    return tuple(
        proj_paths.models / Path(path).with_stem(f'{prefix}_{Path(path).stem}')
        for path in paths
    )

@frozen
class ModelStore:
    model: nn.Module
    scalers: dict[str, Scalers]
    team_features: pd.DataFrame
    config: dict[str, Any]

@frozen
class DataStore:
    X: pd.DataFrame
    teams: pd.DataFrame
    team_encoder: dict[str, int]
    w_store: ModelStore
    nw_store: ModelStore

@st.cache_resource()
def load_resources(wpaths: list[Path], nwpaths: list[Path]) -> DataStore:
    w_model, w_scalers, w_team_features, w_config = load_model(*wpaths)
    w_scalers.pop('final_score_diff', None)
    nw_model, nw_scalers, nw_team_features, nw_config = load_model(*nwpaths)
    nw_scalers.pop('final_score_diff', None)
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
    w_store = ModelStore(w_model, w_scalers, w_team_features, w_config)
    nw_store = ModelStore(nw_model, nw_scalers, nw_team_features, nw_config)
    return DataStore(
        X, teams, team_encoder, w_store, nw_store
    )

data_store = load_resources(get_paths(*wpaths_names, prefix='w'), get_paths(*nwpaths_names, prefix='nw'))


abbrev_away = 'CHI'
abbrev_home = 'CHI'
score_diff = -5

@frozen
class Line:
    x: np.ndarray
    y: np.ndarray

@st.cache_resource()
def init_experiment(abbrev_home: str | None = None, abbrev_away: str | None = None, score_diff: int = 0, _data_store=data_store) -> pd.DataFrame:
    if any([abbrev_home is None, abbrev_away is None]):
        return None
    experiment = prepare_experiment(abbrev_home, abbrev_away, score_diff, _data_store.X.copy(), _data_store.teams) 
    return experiment

st.cache_resource()
def get_plot_values(abbrev_home: str, abbrev_away: str, experiment: pd.DataFrame, _data_store: DataStore) -> Line:
    print("Calculating probabilities...")
    print(experiment[['home_team_id', 'home_team_abbreviation', 'away_team_id', 'away_team_abbreviation', 'home_last_5_pts_diff_avg', 'away_last_5_pts_diff_avg']].iloc[0])
    print(f"{abbrev_home}")
    print(f"{abbrev_away}")
    results = calculate_probs_for_diff(
        abbrev_home, abbrev_away, experiment,
        w_model=_data_store.w_store.model,
        w_scalers=_data_store.w_store.scalers,
        w_config=_data_store.w_store.config, 
        nw_model=_data_store.nw_store.model,
        nw_scalers=_data_store.nw_store.scalers,
        nw_config=_data_store.nw_store.config,
        team_encoder=_data_store.team_encoder
    )
    line = Line(results['time_remaining'], results['probs'])
    return line


if "games" not in st.session_state:
    st.session_state.games = {}
    st.session_state.colors = set()

def create_plot():

    p = figure(title="Win Probability", x_axis_label='Game Time [s]', y_axis_label='P(Home Win)', width=600, height=400)
    for name, game in st.session_state['games'].items():
        p.line(game['x'], game['y'], line_width=2, legend_label=name, color=game['color'])
    p.legend.location = "top_left"
    p.x_range.start = 0
    p.x_range.end = 3000
    p.y_range.start = 0
    p.y_range.end = 1
    # dashed 
    st.bokeh_chart(p)

print("FEATURES")
print(data_store.w_store.team_features + data_store.nw_store.team_features)

if 'experiment' not in st.session_state:
    abbrev_home = 'CHI'
    abbrev_away = 'CHI'
    score_diff = 0
    st.session_state['experiment'] = init_experiment(abbrev_home, abbrev_away, score_diff, data_store)
    for feature in data_store.w_store.team_features + data_store.nw_store.team_features:
        st.session_state[feature] = st.session_state['experiment'][feature].iloc[0].item()

def override_team_data(team_features, play_where, abbrev, data_store) -> dict[str, Any]:
    team_features = [
        col
        for col in team_features
        if col in play_where
    ]
    for feature in team_features:
        st.number_input(
            feature, key=feature, placeholder=0
        )

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

# Sidebar for inputs
with st.sidebar:
    st.header("Add New Matchup")
    team_abbrews = data_store.teams['abbreviation'].to_list()
    st.write("Choose teams:")
    home_team = st.selectbox("Home Team", [""] + team_abbrews)
    override_team_data(data_store.w_store.team_features, 'home', home_team, data_store)
    away_team = st.selectbox("Away Team", [""] + team_abbrews)
    override_team_data(data_store.w_store.team_features, 'away', away_team, data_store)
    score_diff = st.slider("Score Difference: Away - Home", -50, 50, 0, step=1)
    name = st.text_input("Matchup Name", f'{away_team} @ {home_team} = {score_diff}')
    if st.button("Add Game"):
        if name in st.session_state.games:
            st.error("Error: Game name already exists!")
        if len(st.session_state.games) >= game_limit:
            st.error(f"Error: Maximum of {game_limit} games reached! Please delete a game first.")
        elif away_team and home_team:
            experiment = init_experiment(home_team, away_team, score_diff, data_store)
            for feature in data_store.w_store.team_features + data_store.nw_store.team_features:
                experiment[feature] = st.session_state[feature]
            st.session_state['experiment'] = experiment
            line = get_plot_values(
                home_team,
                away_team,
                st.session_state['experiment'],
                data_store
            )
            color = rgb_to_hex(next(color_cycle))
            while color in st.session_state.colors:
                color = rgb_to_hex(next(color_cycle))
            st.session_state.colors.add(color)
            st.session_state.games[name] = {
                'x': line.x,
                'y': line.y,
                'color': color,
                'name': name
            }
            st.success(f"Game '{name}' added successfully!")
        else:
            not_selected = {
                label :team
                for label, team in {
                    'home': home_team,
                    'away': away_team
                }.items()
                if st.session_state.get(f'{team}_team', None) is None
            }
            st.warning(f"Error: {' & '.join(not_selected)} team is not selected!")
    st.header("Reupload the database")
    #if st.button("Import Database"):
    uploaded_file = st.file_uploader("Choose a file", type=['sql'])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        upload_database(uploaded_file)

st.header("Plot & Game Management")
create_plot()

if st.session_state.games:
    selected_games = st.multiselect(
        options=list(st.session_state.games.keys()),
        key="delete_select",
        label="Select games to delete"
    )
    if st.button("Delete Selected Games"):
        for name in selected_games:
            color = st.session_state.games[name]['color']
            del st.session_state.games[name]
            st.session_state.colors.remove(color)
        #status_placeholder.success(f"Deleted {len(selected_games)} games!")
        st.rerun()
else:
    st.info("No games to delete. Add a game first!")

