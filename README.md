# <img src="favicon.ico" width="48"> DataBall: NBA Betting with Machine Learning

[![Test Status](https://github.com/klane/databall/workflows/Tests/badge.svg)](https://github.com/klane/databall/actions)
[![License](https://img.shields.io/github/license/klane/databall.svg?label=License)](LICENSE)

This project collects live NBA betting data from various sources, including sportsbooks like Bet365 and FanDuel. The data is visualized in real-time through a Streamlit dashboard, providing a seamless user experience. Additionally, we utilize historical NBA game data to train a logistic regression model on approximately 13 million records (2 GB of data), achieving an 86% performance accuracy.

Contents:

- [covers](https://github.com/klane/databall/tree/main/databall/covers): Scrapy project to scrape point spreads and over/under lines from [covers.com](http://covers.com)
- [databall](https://github.com/klane/databall/tree/main/databall): Python module with support functions to perform tasks including collecting stats to a SQLite database, simulating seasons, and customizing plots
- [docs](https://github.com/klane/databall/tree/main/docs): Code required to build the GitHub Pages [site](https://klane.github.io/databall/) for this project
- [notebooks](https://github.com/klane/databall/tree/main/notebooks): Jupyter notebooks of all analyses
- [report](https://github.com/klane/databall/tree/main/report): LaTeX files for report and slides

Link to a test database with data from 1990 - March 2020 [test nba.db file](https://drive.google.com/file/d/10CBcCLv2N_neFL39ThykcudUVUv5xqLB/view?usp=sharing)
