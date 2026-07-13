from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    debug: bool = False
    secret_key: str
    allowed_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173"]
    enable_scheduler: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

settings = Settings() # type:ignore