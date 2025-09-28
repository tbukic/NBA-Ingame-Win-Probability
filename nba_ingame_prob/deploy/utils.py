import pandas as pd
from attrs import frozen


@frozen(slots=True)
class Line:
    home_team: str
    away_team: str
    data: pd.DataFrame
    score_diff: float
