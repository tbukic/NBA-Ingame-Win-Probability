import pandas as pd # type: ignore
from sbrscrape import Scoreboard # type: ignore
from datetime import datetime


class SbrOddsProvider:
    def __init__(self):  # Format date to 'YYYY-MM-DD'
        sb = Scoreboard(sport="NBA")  # Specify the date for today's games
        self.games = sb.games if hasattr(sb, 'games') else []
        self.sportbooks = ["fanduel", "betmgm", "caesars", "draftkings", "bet365"]  # All sportsbooks

    def moneyline_to_implied_probability(self, moneyline):
        """Convert moneyline odds to implied probability (price between 0 and 1)."""
        if moneyline > 0:
            # Positive moneyline odds
            return 100 / (moneyline + 100)
        else:
            # Negative moneyline odds
            return -moneyline / (-moneyline + 100)

    def get_odds(self):
        """Retrieve odds from the games and return as a dictionary."""
        dict_res = {}
        for game in self.games:
            home_team_name = game['home_team'].replace("Los Angeles Clippers", "LA Clippers")
            away_team_name = game['away_team'].replace("Los Angeles Clippers", "LA Clippers")

            game_odds = {
                'under_over_odds': game['total'].get(self.sportbooks[0], None),  # Default to the first sportsbook
                home_team_name: {},
                away_team_name: {}
            }

            # Loop through all sportsbooks and get the moneyline, then convert to implied probability (price)
            for sportsbook in self.sportbooks:
                home_moneyline = game['home_ml'].get(sportsbook, None)
                away_moneyline = game['away_ml'].get(sportsbook, None)

                # Convert moneyline to implied probability (price)
                home_price = self.moneyline_to_implied_probability(home_moneyline) if home_moneyline is not None else None
                away_price = self.moneyline_to_implied_probability(away_moneyline) if away_moneyline is not None else None

                game_odds[home_team_name][sportsbook] = home_price
                game_odds[away_team_name][sportsbook] = away_price

            dict_res[home_team_name + ':' + away_team_name] = game_odds

        return dict_res

    def get_odds_table(self):
        """Convert the odds dictionary into a structured table using pandas."""
        odds = self.get_odds()
        table_data = []

        # Group data by teams
        teams_data = {}

        for game, details in odds.items():
            home_team, away_team = game.split(":")

            if home_team not in teams_data:
                teams_data[home_team] = {sportsbook: None for sportsbook in self.sportbooks}
            if away_team not in teams_data:
                teams_data[away_team] = {sportsbook: None for sportsbook in self.sportbooks}

            # Set the implied probability (price) for each team for each sportsbook
            for sportsbook in self.sportbooks:
                teams_data[home_team][sportsbook] = details[home_team].get(sportsbook, None)
                teams_data[away_team][sportsbook] = details[away_team].get(sportsbook, None)

        # Prepare rows for the table
        for team, odds in teams_data.items():
            row = {"Team": team}
            row.update(odds)
            table_data.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(table_data)
        return df
