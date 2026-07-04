from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    hf_token: str = ""
    use_overlap_heuristic_fallback: bool = False
    
    worker_pool_size: int = 2
    request_timeout_seconds: int = 180
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
