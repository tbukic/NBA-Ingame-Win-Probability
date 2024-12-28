import joblib
import pandas as pd
import streamlit as st

from nba_api.live.nba.endpoints import scoreboard
from sklearn.preprocessing import LabelEncoder

from nba_betting_ai.utils.utils import parse_time, convert_time_to_seconds


@st.cache_resource
def load_model(model_path="Model/multi_output_model.pkl"):
    """
    Loads the pre-trained MultiOutputClassifier model and caches it.

    Parameters:
    - model_path: Path to the saved pre-trained model.

    Returns:
    - model: Loaded MultiOutputClassifier model.
    """
    model = joblib.load(model_path)
    return model



def extract_game_data():
    """
    Fetches today's NBA scoreboard data using nba_api and extracts relevant game data
    into a DataFrame with columns: HOME TEAM, AWAY TEAM, TIME REMAINING, POINTS DIFF.

    Returns:
        pd.DataFrame: DataFrame with columns HOME TEAM, AWAY TEAM, TIME REMAINING, POINTS DIFF.
    """
    # Fetch the scoreboard data
    games = scoreboard.ScoreBoard()
    games_data = games.get_dict()['scoreboard']['games']  # Access the games list

    # Initialize a list to store extracted data
    extracted_data = []

    # Loop through each game and extract details
    for game in games_data:
        period = game['period']
        team1 = game['homeTeam']['teamTricode']
        team2 = game['awayTeam']['teamTricode']
        team1_score = game['homeTeam']['score']
        team2_score = game['awayTeam']['score']
        time_remaining = game.get('gameClock', 'N/A')  # Get time remaining (e.g., 'PT05M20.00S' or '5:20')

        # Parse the time remaining if it's in the expected format
        time_remaining = parse_time(time_remaining)

        # Append the data
        extracted_data.append({
            'Period': period,
            'Time': time_remaining,
            'AwayName': team2,
            'AwayScore': team2_score,
            'HomeName': team1,
            'HomeScore': team1_score,
        })

    # Convert to DataFrame
    df = pd.DataFrame(extracted_data)
    return df


def trans_data(df):
    """
    Preprocesses the input dataframe, applies transformations, and uses a pre-trained MultiOutputClassifier model 
    to predict HomeWin and AwayWin outcomes.

    Parameters:
    - df: DataFrame with the dataset to be processed.

    Returns:
    - predictions_df: DataFrame with predictions and probabilities.
    """

    df['TimeInSeconds'] = df['Time'].apply(convert_time_to_seconds)
    df = df.dropna()
    df['TimeRemaining'] = (4 - df['Period']) * 12 * 60 + df['TimeInSeconds']

    # Encode categorical columns (AwayName and HomeName)
    label_encoder = LabelEncoder()
    df['AwayName_encoded'] = label_encoder.fit_transform(df['AwayName'])
    df['HomeName_encoded'] = label_encoder.fit_transform(df['HomeName'])

    #Make sure the data is live data
    df = df.dropna(subset=['TimeInSeconds'])
    
    if len(df) > 0:
        # Drop the original categorical columns and unnecessary columns
        df_encoded = df.drop(['AwayName', 'HomeName', 'Time', 'Period'], axis=1)

        # Prepare features (X)
        X = df_encoded

        # Load the model (cached)
        model = load_model()

        # Predict on the entire dataframe for both HomeWin and AwayWin
        y_pred = model.predict(X)

        # Get the predicted probabilities for both HomeWin and AwayWin
        y_prob = model.predict_proba(X)

        # Extract the probabilities for the 'win' class (1) for both HomeWin and AwayWin
        home_win_prob = y_prob[0][:, 1]  # Home win probability
        away_win_prob = y_prob[1][:, 1]  # Away win probability

        # Prepare the dataframe for results
        df_pred = X.copy()

        # Add the predicted probabilities to the dataframe
        df_pred['predicted_home_win_probability'] = home_win_prob
        df_pred['predicted_away_win_probability'] = away_win_prob

        # Map the encoded columns back to their original names
        df_pred['AwayName'] = label_encoder.inverse_transform(df_pred['AwayName_encoded'])
        df_pred['HomeName'] = label_encoder.inverse_transform(df_pred['HomeName_encoded'])

        # Return the final dataframe with predictions and probabilities
        predictions_df = df_pred[['AwayName', 'HomeName', 'predicted_home_win_probability', 'predicted_away_win_probability']]

        return predictions_df
    
    return None