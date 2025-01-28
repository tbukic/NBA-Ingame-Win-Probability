from attrs import frozen
from pathlib import Path

project_root = Path(__file__).parent.parent

@frozen
class ConfigPath:
    folder: Path = project_root / "config"
    default: Path = folder / "run_config.yaml"

@frozen
class ProjectPaths:
    logs: Path = project_root / "logs"
    pg_dump: Path = project_root / "pg_dump"
    output: Path = project_root / "output"
    models: Path = project_root / "models"
    config: ConfigPath = ConfigPath()

@frozen
class GameInfo:
    period_min = 12
    periods_regular = 4
    sec_min= 60
    match_time = periods_regular*period_min*sec_min


proj_paths = ProjectPaths()
game_info = GameInfo()
