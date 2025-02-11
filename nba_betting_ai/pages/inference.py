import altair as alt
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
from nba_betting_ai.deploy.inference import calculate_probs_for_diff, load_model, prepare_experiment, Line
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

if not st.session_state.get('database_ready', check_database_exists()):
    with st.sidebar:
        st.button("Upload database", on_click=prompt_upload_database)

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
    model: nn.Module
    scalers: dict[str, Scalers]
    team_features: pd.DataFrame
    config: dict[str, Any]

@frozen
class DataStore:
    X: pd.DataFrame
    teams: pd.DataFrame
    team_encoder: dict[str, int]
    store: ModelStore

@st.cache_resource()
def load_resources(model_path: Path, model_init_path: Path, config_path: Path, scalers_path: Path) -> DataStore:
    model, scalers, team_features, config = load_model(model_path, model_init_path, config_path, scalers_path)
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

def select_model():
    model_name = st.session_state.model
    model_path = next(proj_paths.models.glob(f'*{model_name}*'))
    model_descr = model_name.split('-', 1)[1]
    model_init_path = model_path.with_suffix('.yaml').with_stem(f'model_init-{model_descr}')
    config_path = model_path.with_suffix('.yaml').with_stem(f'run_config-{model_descr}')
    scalers_path = model_path.with_suffix('.pkl').with_stem(f'scalers-{model_descr}')
    st.session_state.data_store = load_resources(model_path, model_init_path, config_path, scalers_path)

model_options = sorted(
    model.stem
    for model in proj_paths.models.glob('*.pth')
)

with st.sidebar:
    st.header("Select model")
    st.selectbox(
        "Model",
        key='model',
        options=model_options,
        on_change=select_model
    )
    if not 'data_store' in st.session_state:
        select_model()

def select_team(side: str):
    # st.write(f"{side.title()} team data:")
    data_store = st.session_state.data_store
    st.selectbox(
        f"{side.title()} Team",
        key=f'{side}_team',
        options=data_store.teams['abbreviation'].tolist(),
        placeholder=f"Pick a {side} team"        
    )
    team_features = [
        feature
        for feature in data_store.store.team_features
        if side in feature
    ]
    for feature in team_features:
        step = 0.2 if 'avg' in feature else 1
        st.number_input(
            feature,
            key=feature,
            placeholder=0,
            step=step,
            format="%.1f" if 'avg' in feature else "%d"
        )

if not 'matchups' in st.session_state:
    st.session_state.matchups = {}
if not 'plot_colors' in st.session_state:
    st.session_state.plot_colors = {}
if not 'plot_data' in st.session_state:
    st.session_state.plot_data = {}

@st.cache_data
def get_matchup_data(home_abbrev, away_abbrev):
    data_store = st.session_state.data_store
    experiment = prepare_experiment(home_abbrev, away_abbrev, score_diff, data_store.X.copy(), data_store.teams)
    return experiment

def generate_matchup_name(home_abbrev, away_abbrev) -> str:
    simple_name = f'{away_abbrev} @ {home_abbrev}'
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

@st.cache_data
def get_plot_data(home_abbrev, away_abbrev, experiment: pd.DataFrame, score_diff: float) -> Line:
    experiment = experiment.copy()
    experiment['score_diff'] = score_diff
    data_store = st.session_state.data_store
    results = calculate_probs_for_diff(
        abbrev_home=st.session_state.home_team,
        abbrev_away=st.session_state.away_team,
        experiment=experiment,
        model=data_store.store.model,
        scalers=data_store.store.scalers,
        team_features=data_store.store.team_features, 
        team_encoder=data_store.team_encoder
    )
    line = Line(home_abbrev, away_abbrev, tuple(results['time_remaining']), tuple(results['probs']), score_diff)
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
        st.session_state.plot_data[name] = get_plot_data(home_abbrev, away_abbrev, experiment, st.session_state.score_diff)
        st.rerun()

def add_matchup():
    for team in ('home', 'away'):
        if st.session_state.get(f'{team}_team') is None:
            st.errorn(f"Please select {team} team")
            return
    experiment = get_matchup_data(st.session_state.home_team, st.session_state.away_team)
    data_store = st.session_state.data_store
    for feature in data_store.store.team_features:
        experiment[feature] = st.session_state[feature]
    add_named_matchup(experiment)

def select_teams():
    with st.form(key='team_selection'):
        select_team('away')
        select_team('home')
        st.form_submit_button("Add matchup", on_click=add_matchup)

def draw_plots():
    p = figure(title="Win Probability", x_axis_label='Game Time [s]', y_axis_label='P(Home Win)', width=600, height=400)
    for name, game in st.session_state.plot_data.items():
        p.line(game.x, game.y, line_width=2, legend_label=name, color=st.session_state.plot_colors[name])
    p.legend.location = "top_left"
    p.x_range.start = 0
    p.x_range.end = 3000
    p.y_range.start = 0
    p.y_range.end = 1
    st.bokeh_chart(p)

@st.dialog("Scrape data")
def scrape_new_data():
    st.write("Scraping data...")
    # scrape_everything(
    #     engine=engine,
    #     season='2024-25',
    #     start_date=None,
    #     end_date=None,
    #     headers=None
    # )
    st.rerun()

def delete_selected_games():
    for game in st.session_state.delete_select:
        st.session_state.matchups.pop(game)
        st.session_state.plot_colors.pop(game)
        st.session_state.plot_data.pop(game)
    st.rerun()

def reparametrize_plots():
    st.session_state.plot_data = {
        matchup: get_plot_data(data.home_team, data.away_team, st.session_state.matchups[matchup], st.session_state.score_diff)
        for matchup, data in st.session_state.plot_data.items()
    }
    # st.rerun()

with st.sidebar:
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