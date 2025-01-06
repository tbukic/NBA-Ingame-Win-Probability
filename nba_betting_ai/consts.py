from attrs import frozen
from pathlib import Path

project_root = Path(__file__).parent.parent

@frozen
class ProjectPaths:
    logs = project_root / "logs"
    pg_dump = project_root / "pg_dump"


proj_paths = ProjectPaths()