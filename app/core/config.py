from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    debug:bool = False
    secret_key: str
    
    model_config = SettingsConfigDict(env_file=".env", extra="allow")
    
settings = Settings() # type:ignore