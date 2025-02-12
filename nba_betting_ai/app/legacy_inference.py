import pandas as pd
import random
import streamlit as st

from nba_betting_ai.model.prediction import trans_data

# List of all 30 NBA teams with their 3-letter abbreviations
team_list = [
    'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 
    'CLE', 'DAL', 'DEN', 'DET', 'GSW', 
    'HOU', 'IND', 'LAC', 'LAL', 'MEM', 
    'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 
    'OKC', 'ORL', 'PHI', 'PHX', 'POR', 
    'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
]

# Streamlit UI components
st.title('Dynamic Basketball Score Tracker with Prediction')

# Dropdown for selecting the period
period = st.selectbox('Select Period', [1, 2, 3, 4], index=0)

# Time input field
time = st.text_input('Enter Time (e.g., 12:00)', value='12:00')

# Dropdown for selecting Away and Home teams
away_name = st.selectbox('Select Away Team', team_list, index=0)

# Ensure home team is not the same as away team
home_name = st.selectbox(
    'Select Home Team', 
    [team for team in team_list if team != away_name], 
    index=1
)

# Initialize session state for scores if not already initialized
if 'away_score' not in st.session_state:
    st.session_state['away_score'] = random.randint(80, 120)

if 'home_score' not in st.session_state:
    st.session_state['home_score'] = random.randint(80, 120)

# Score input fields - Allow the user to directly input the score as a text field
away_score_input = st.text_input(f'Away Team ({away_name}) Score', value=str(st.session_state['away_score']))
home_score_input = st.text_input(f'Home Team ({home_name}) Score', value=str(st.session_state['home_score']))

# Validate input scores - Ensure that both are numeric
try:
    away_score = int(away_score_input)
    home_score = int(home_score_input)
    
    # Update session state with new scores
    st.session_state['away_score'] = away_score
    st.session_state['home_score'] = home_score
except ValueError:
    st.error("Please enter valid numeric scores for both teams.")
    away_score = None
    home_score = None

# Create a DataFrame with the inputs
if away_score is not None and home_score is not None:
    game_data = pd.DataFrame({
        'Period': [period],
        'Time': [time],
        'AwayName': [away_name],
        'AwayScore': [away_score],
        'HomeName': [home_name],
        'HomeScore': [home_score]
    })

    # Display the game data
    st.subheader('Game Data:')
    st.table(game_data)

    # Store the team names before prediction
    stored_away_name = away_name
    stored_home_name = home_name

    # Display prediction logic (for simplicity, this can be replaced with your actual model)
    if st.button('Predict Winner'):
        # Make sure the data passed to trans_data is correct
        st.write("Input Data to Model:", game_data)

        # Call the prediction function (ensure the model function handles this correctly)
        result = trans_data(game_data) 

        # Display the prediction
        st.subheader('Prediction')

        # Ensure the result shows the correct team names (using the stored values)
        if 'AwayName' in result.columns and 'HomeName' in result.columns:
            # Reassign the correct names if needed (to fix the name issue in the prediction dataframe)
            result['AwayName'] = stored_away_name
            result['HomeName'] = stored_home_name

            # Display the result with the correct names
            st.table(result[['AwayName', 'HomeName', 'predicted_home_win_probability', 'predicted_away_win_probability']])
        else:
            st.error("Prediction output doesn't contain expected columns.")
