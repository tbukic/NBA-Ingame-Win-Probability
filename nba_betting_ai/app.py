import streamlit as st # type: ignore
import time

from nba_betting_ai.model.prediction import trans_data, extract_game_data
from nba_betting_ai.utils.data_providers import SbrOddsProvider
from nba_betting_ai.utils.utils import moneyline_data, moneyline_to_probability, nba_live_data

# Set Streamlit page config for better visuals
st.set_page_config(page_title="NBA Live Scores", page_icon="🏀", layout="wide")


# Center the title of the app
st.markdown(
    """
    <style>
        .centered-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #ffffff;
        }
    </style>
    <div class="centered-title">
        🏀 NBA Live Scores
    </div>
    """,
    unsafe_allow_html=True,
)

# Subtitle
st.markdown(
    """
    <div style="text-align: center; color: #999; font-size: 16px; margin-bottom: 20px;">
        Stay updated with <strong>real-time NBA game updates</strong>! Data refreshes every minute.
    </div>
    """,
    unsafe_allow_html=True,
)


# Display live NBA data
def display_nba_live_data():
    placeholder = st.empty()  # Placeholder for dynamic table updates

    while True:  # Continuous refresh
        # Fetch live data
        games_data = nba_live_data()
        moneyline_df = moneyline_data()

        games = games_data.get("scoreboard", {}).get("games", [])

        #Fetch sportbooks data 
        sbr_provider = SbrOddsProvider()
        odds_df = sbr_provider.get_odds_table()
        money_line_data = odds_df[['Team', 'bet365']]

        #test ML model 
        ml_data = extract_game_data()
        result = trans_data(ml_data)
        

        # Update the placeholder content
        with placeholder.container():
            if games:
                for game in games:
                    home_team = game["homeTeam"]["teamName"]
                    away_team = game["awayTeam"]["teamName"]
                    home_score = game["homeTeam"]["score"]
                    away_score = game["awayTeam"]["score"]
                    status = game["gameStatusText"]
                    home_full_name = f"{game['homeTeam']['teamCity']} {home_team}"
                    away_full_name = f"{game['awayTeam']['teamCity']} {away_team}"
                    home_teamTricode = game['homeTeam']['teamTricode']
                    away_teamTricode = game['awayTeam']['teamTricode']

                    st.write()
                    #home_moneyline_array = money_line_data[money_line_data["Team"] == home_full_name]["bet365"].values
                    #away_moneyline_array = money_line_data[money_line_data["Team"] == away_full_name]["bet365"].values

                    home_moneyline_array = moneyline_df[moneyline_df["home"] == home_full_name]["odds.home.moneyLine"].values
                    away_moneyline_array = moneyline_df[moneyline_df["away"] == away_full_name]["odds.away.moneyLine"].values
                    
                    # Extract scalar values if they exist
                    home_moneyline = round(float(home_moneyline_array[0]), 2) if len(home_moneyline_array) > 0 else None
                    away_moneyline = round(float(away_moneyline_array[0]), 2) if len(away_moneyline_array) > 0 else None
                    
                    if home_moneyline and away_moneyline:
                        home_price = moneyline_to_probability(home_moneyline)
                        away_price = moneyline_to_probability(away_moneyline)

                    else: home_price, away_price = None, None    


                    st.markdown(
                        f"""
                        <div style="border: 2px solid #444; border-radius: 15px; padding: 20px; margin-bottom: 20px; background: linear-gradient(145deg, #0f0f0f, #202020); color: #ffffff; max-width: 700px; margin-left: auto; margin-right: auto; box-shadow: 3px 3px 8px #000000, -3px -3px 8px #2a2a2a;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="text-align: left; width: 45%;">
                                    <h3 style="margin: 10px 0; color: #4CAF50;">{home_full_name}</h3>
                                    <p style="margin: 5px 0; font-size: 16px; color: #dddddd;"> Score: <strong>{home_score}</strong></p>
                                    <p style="margin: 5px 0; font-size: 16px; color: #dddddd;"> Price: <strong>{home_price}</strong></p>
                                    <p style="margin: 5px 0; font-size: 16px; color: #dddddd;"> Win Probability: <strong>{f"{result[result['HomeName'] == home_teamTricode]['predicted_home_win_probability'].iloc[0] * 100:.1f}%" if result is not None and not result[result['HomeName'] == home_teamTricode].empty else "0.0%"}</strong></p>
                                </div>
                                <div style="text-align: center; font-size: 16px; color: #bbbbbb; width: 10%;">
                                    <p style="margin: 0; font-weight: bold; color: #FFD700;">{status}</p>
                                </div>
                                <div style="text-align: right; width: 45%;">
                                    <h3 style="margin: 10px 0; color: #FF5722;">{away_full_name}</h3>
                                    <p style="margin: 5px 0; font-size: 16px; color: #dddddd;"> Score: <strong>{away_score}</strong></p>
                                    <p style="margin: 5px 0; font-size: 16px; color: #dddddd;"> Price: <strong>{away_price}</strong></p>
                                    <p style="margin: 5px 0; font-size: 16px; color: #dddddd;"> Win Probability: <strong>{f"{result[result['HomeName'] == home_teamTricode]['predicted_away_win_probability'].iloc[0] * 100:.1f}%" if result is not None and not result[result['HomeName'] == home_teamTricode].empty else "0.0%"}</strong></p>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Sportsbook tables
                st.markdown(
                    """
                    <div style="text-align: center; font-size: 20px; margin-top: 20px; color: #FFD700;">
                         <strong>Sportsbook Odds</strong> 
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.table(odds_df)


            else:
                st.markdown(
                    """
                    <div style="text-align: center; margin-top: 50px; color: #ffffff; font-size: 22px;">
                        <h4>🚫 No live NBA games available right now. 🚫</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        time.sleep(60)


if __name__ == "__main__":
    display_nba_live_data()



# ANOTHER SOURCE OF DATA
# home_moneyline = moneyline_df[moneyline_df["home"] == home_full_name]["odds.home.moneyLine"].values
# away_moneyline = moneyline_df[moneyline_df["away"] == away_full_name]["odds.away.moneyLine"].values