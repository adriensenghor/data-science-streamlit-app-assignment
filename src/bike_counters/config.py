from dataclasses import dataclass


@dataclass
class AppConfig:
    data_dir: str = "data"
    env: str = "dev"
