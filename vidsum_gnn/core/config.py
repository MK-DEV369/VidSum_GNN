import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "VIDSUM-GNN"
    API_V1_STR: str = "/api"
    
    # Database
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "vidsum")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ASR (Whisper)
    # - "auto": let Whisper detect the spoken language
    # - ISO language codes like "en", "es", "hi", ... to force a specific language
    WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "auto")

    # Storage
    UPLOAD_DIR: str = os.path.join(os.getcwd(), "data", "uploads")
    PROCESSED_DIR: str = os.path.join(os.getcwd(), "data", "processed")
    OUTPUT_DIR: str = os.path.join(os.getcwd(), "data", "outputs")
    TEMP_DIR: str = os.path.join(os.getcwd(), "data", "temp")
    MODEL_DIR: str = os.path.join(os.getcwd(), "model", "models")
    # Primary GNN checkpoint (binary model)
    GNN_CHECKPOINT: str = os.path.join(os.getcwd(), "model", "models", "checkpoints", "vidsum_gnn_best_binary.pt")
    # Model Defaults
    CHUNK_DURATION: int = 300  # 5 minutes
    CHUNK_OVERLAP: int = 30    # 30 seconds overlap
    
    # GNN Model
    GNN_HIDDEN_DIM: int = 1024
    GNN_NUM_HEADS: int = 8
    GNN_NUM_LAYERS: int = 2
    # Inference behavior
    DECISION_THRESHOLD: float = 0.5
    TOPK_RATIO: float = 0.15  # fraction of shots to select for summarization
    
    class Config:
        case_sensitive = True

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)

print(f"✓ Configuration loaded:")
print(f"  - Database: {settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")
print(f"  - Log Level: {settings.LOG_LEVEL}")
print(f"  - Upload Dir: {settings.UPLOAD_DIR}")
print(f"  - Chunk Duration: {settings.CHUNK_DURATION}s")
