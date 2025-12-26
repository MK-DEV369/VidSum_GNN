from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from typing import Dict
import json

from vidsum_gnn.db.models import Base
from vidsum_gnn.core.config import settings
from vidsum_gnn.api.routes import router
from vidsum_gnn.utils import setup_logging

# Setup logging
setup_logging(log_level=settings.LOG_LEVEL)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, video_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[video_id] = websocket
        print(f"✓ WebSocket connected for video: {video_id}")
    
    def disconnect(self, video_id: str):
        if video_id in self.active_connections:
            del self.active_connections[video_id]
            print(f"✓ WebSocket disconnected for video: {video_id}")
    
    async def send_log(self, video_id: str, log_data: dict):
        if video_id in self.active_connections:
            try:
                await self.active_connections[video_id].send_text(json.dumps(log_data))
            except Exception as e:
                print(f"✗ Failed to send log to {video_id}: {e}")
                self.disconnect(video_id)

manager = ConnectionManager()

# Create async engine and session factory
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=20,
    max_overflow=0,
)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓ Database tables created")
    print(f"✓ VidSum GNN API started on port 8000")
    print(f"✓ Docs available at http://localhost:8000/docs")
    yield
    # Shutdown: Close engine
    await engine.dispose()
    print("✓ Database connection closed")

app = FastAPI(
    title="VidSum GNN API",
    description="Video Summarization with Graph Neural Networks",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {
        "message": "VidSum GNN API is running",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "database": "connected"}

@app.websocket("/ws/logs/{video_id}")
async def websocket_logs(websocket: WebSocket, video_id: str):
    """
    WebSocket endpoint for streaming real-time logs to frontend.
    """
    await manager.connect(video_id, websocket)
    try:
        while True:
            # Keep connection alive and receive any messages
            data = await websocket.receive_text()
            # Echo back confirmation
            await websocket.send_json({"status": "connected", "video_id": video_id})
    except WebSocketDisconnect:
        manager.disconnect(video_id)
        print(f"✓ Client disconnected from video: {video_id}")

# Export manager for use in other modules
__all__ = ["app", "async_session", "manager"]
