# NBA In-Game Win Probability

In-game **home win probability** for NBA based on time remaining, score differential, and team form.  
Models: Ridge/Logistic, CatBoost (with uncertainty), Bayesian-style NN.  
Interactive demo via Streamlit.

**Full write-up:** [`paper/report.pdf`](paper/report.pdf)

## Quick start
Adjust mount paths and environment variables in  `.devcontainer/docker-compose.yml` and open projet in [Dev Container](https://code.visualstudio.com/docs/devcontainers/containers).

```bash
# 1) run a local development version
poe dev

# 2) run the app in docker container
poe app
```

# Application

![Example image](images/04-matchup-score5.png)