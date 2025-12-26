from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text
from vidsum_gnn.core.config import settings
from vidsum_gnn.db.models import Base

# Create engine and session factory
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=20,
    max_overflow=0,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Alias for backward compatibility
AsyncSessionLocal = async_session_factory

async def get_db() -> AsyncSession:
    async with async_session_factory() as session:
        yield session

async def init_db():
    async with engine.begin() as conn:
        # Create tables
        await conn.run_sync(Base.metadata.create_all)
        
        # Convert shots to hypertable (TimescaleDB specific)
        # We wrap in try/except block in case it's already a hypertable or extension missing
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
        except Exception as e:
            print(f"Warning: Could not create timescaledb extension (might need superuser): {e}")

        try:
            # Check if hypertable exists first to avoid error
            await conn.execute(text("SELECT create_hypertable('shots', 'created_at', if_not_exists => TRUE);"))
        except Exception as e:
            print(f"Warning: Could not create hypertable (might not be using TimescaleDB or already exists): {e}")
