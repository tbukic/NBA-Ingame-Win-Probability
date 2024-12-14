# <img src="favicon.ico" width="48"> DataBall: NBA Betting with Machine Learning

[![Test Status](https://github.com/klane/databall/workflows/Tests/badge.svg)](https://github.com/klane/databall/actions)
[![License](https://img.shields.io/github/license/klane/databall.svg?label=License)](LICENSE)

This project collects live NBA betting data from various sources, including sportsbooks like Bet365 and FanDuel. The data is visualized in real-time through a Streamlit dashboard, providing a seamless user experience. Additionally, we utilize historical NBA game data to train a logistic regression model on approximately 13 million records (2 GB of data), achieving an 86% performance accuracy.

## Contents:

- [DataProviders](https://github.com/Younes1337/nba-betting-ai/blob/main/utils/DataProviders.py): Scrape Data from various sportsbooks like [bet365.com](https://www.bet365.com/#/HO/)
- [utils](https://github.com/Younes1337/nba-betting-ai/blob/main/utils/utils.py): Python utils functions (Helpers) that help to calculate or convert metrics.
- [Inference Model](https://github.com/Younes1337/nba-betting-ai/blob/main/pages/inference_model.py): This Python streamlit page contains a UI to allow the user to test the model with a custom input.
- [notebook](https://github.com/Younes1337/nba-betting-ai/blob/main/Model/NBA%20Model.ipynb): Jupyter notebook that contains the process of developing the logistic regression model.
- [Main APP](https://github.com/Younes1337/nba-betting-ai/blob/main/APP.py): The Main App that contains the logic used in the application.

## Architecture
That's an overview of the system, first, we scrape live moneyline data from NBA ESPN for the ongoing games, and also scrape other live moneyline data from different sportsbooks using Python scrapping package and Pandas to manipulate the data besides a real-time machine learning model that predict for each game the team's win probability, the model developed using Skiti-learn which is Logistic regression model, you can find the model here [Logistic Regression Model](https://github.com/Younes1337/nba-betting-ai/blob/main/Model/multi_output_model.pkl), then we used a streamlit UI to make a seamless dashboard for the live data also a page for model inference to test the model on a custom input.
<img src="Arch.png"> 

here's a dashboard example of a real-time NBA games : it represents live games , and the score, price, and the win probability of each team
<p align="center">
  <img src="real-time-nba-game.png" alt="NBA Game">
</p>



Link to a test database with data from 1990 - March 2020 [test nba.db file](https://drive.google.com/file/d/10CBcCLv2N_neFL39ThykcudUVUv5xqLB/view?usp=sharing)
