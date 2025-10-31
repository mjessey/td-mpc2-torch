import tomllib
from pydantic import BaseModel, PositiveInt


class Config(BaseModel):
    buffer_size: PositiveInt
    min_history: PositiveInt
    batch_size: PositiveInt


def read_cfg(cfg_path: str | None = None) -> Config:
    if cfg_path is None:
        cfg_path = "./cfg/default.toml"

    with open(cfg_path, "rb") as f:
        data = tomllib.load(f)

    cfg = Config.model_validate(data["config"])

    return cfg
