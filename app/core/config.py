from typing import Annotated

from pydantic import Field, field_validator
from app.ml.recommendations import RecommendationPolicy
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

def site_pump_policies():
    """Agreed prototype thresholds for this installation only."""
    return {2: RecommendationPolicy(
        sensor_id=4, minimum_level_m=0.5, consideration_level_m=1.0,
        basis="Operator-selected minimum of 0.5 m above the fixed sensor, with a 0.5 m consideration margin. Advisory prototype thresholds, not a validated safe yield or pumping duration.",
    )}


class Settings(BaseSettings):
    database_url: str
    debug: bool = False
    secret_key: str
    allowed_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173"]
    enable_scheduler: bool = True
    level_model_path: str = "models/level_change_linear.json"

    pump_recommendation_policies: dict[int, RecommendationPolicy | None] = Field(default_factory=site_pump_policies)

    @field_validator("pump_recommendation_policies")
    @classmethod
    def _site_policy_with_overrides(cls, policies):
        # An empty deployment mapping retains the agreed site policy.
        # Explicit {"2": null} disables advice for this well.
        return {**site_pump_policies(), **policies}

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

settings = Settings() # type:ignore