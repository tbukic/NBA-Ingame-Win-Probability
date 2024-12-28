import bs4 # type: ignore
from bs4 import BeautifulSoup # type: ignore
import requests # type: ignore
import pandas as pd # type: ignore
import pytz # type: ignore
from datetime import datetime
from nba_api.live.nba.endpoints import scoreboard # type: ignore
import re


def extract_money_lines(url, dateStr):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}
    payload = {
        'sport': 'basketball',
        'league': 'nba',
        'region': 'us',
        'lang': 'en',
        'contentorigin': 'espn',
        'buyWindow': '1m',
        'showAirings': 'buy,live,replay',
        'tz': 'America/New_York',
        'dates': dateStr}

    response = requests.get(url, headers=headers, params=payload).json()
    events = response['sports'][0]['leagues'][0]['events']

    df = pd.json_normalize(events,
                           record_path=['competitors'],
                           meta=['odds', ['odds', 'away', 'moneyLine'], ['odds', 'home', 'moneyLine']],
                           errors='ignore')

    reshaped_data = []

    for i in range(0, len(df), 2):
        row_away = df.iloc[i]
        row_home = df.iloc[i + 1]
        reshaped_row = {
            'away': f"{row_away['displayName']}",
            'home': f"{row_home['displayName']}",
            'odds.away.moneyLine': row_away['odds.away.moneyLine'],
            'odds.home.moneyLine': row_home['odds.home.moneyLine']
        }
        reshaped_data.append(reshaped_row)

    reshaped_df = pd.DataFrame(reshaped_data)

    return reshaped_df


def moneyline_to_probability(moneyline):
    """
    Converts a moneyline to its implied probability (in decimal form).
    """
    try:
        if moneyline > 0:
            probability = 100 / (moneyline + 100)
        else:
            probability = abs(moneyline) / (abs(moneyline) + 100)
        return round(probability, 4)  # Return a decimal probability (e.g., 0.9091)
    except ValueError:
        return "N/A"  # Handle cases where the moneyline is invalid




def moneyline_data():
    usa_timezone = pytz.timezone('America/New_York')
    current_time_usa = datetime.now(usa_timezone)
    dateStr = current_time_usa.strftime('%Y%m%d')
    url = "https://site.web.api.espn.com/apis/v2/scoreboard/header"
    df = extract_money_lines(url, dateStr)
    return df


def nba_live_data():
    games = scoreboard.ScoreBoard()
    data = games.get_dict()
    return data

def convert_time_to_seconds(time_str):
    try:
        minutes, seconds = time_str.split(':')
        return int(minutes) * 60 + int(seconds)
    except ValueError:
        # Handle invalid time formats or missing values
        return None

def parse_time(duration_str):
  """
  Converts duration formats (ISO 8601, MM:SS, or HH:MM) to 'min:s' format.

  Args:
      duration_str (str): The duration string to parse.

  Returns:
      str: The parsed duration in 'min:s' format or "N/A" if format is unsupported.
  """
  # Check for ISO 8601 duration format (PT05M20.00S)
  match = re.match(r'PT(\d+)M(\d+(\.\d+)?)S', duration_str)
  if match:
    minutes = int(match.group(1))
    seconds = float(match.group(2))
    return f"{minutes}:{int(seconds)}"

  # Handle MM:SS format
  match = re.match(r'(\d+):(\d+)', duration_str)
  if match:
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    return f"{minutes}:{seconds}"

  # Handle HH:MM format (hours and minutes)
  match = re.match(r'(\d+):(\d+)', duration_str)
  if match:
    hours = int(match.group(1))
    minutes = int(match.group(2))
    total_minutes = hours * 60 + minutes
    return f"{total_minutes}:00"  # Assuming seconds are 0 for HH:MM format

  # Return "N/A" for unsupported formats
  return "N/A"