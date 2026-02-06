# VIDSUM-GNN Quick Start Guide

## Prerequisites
- Docker Desktop (with Docker Compose)
- Git
- ~20GB free disk space (for models and data)

## Installation & Running

### Option 1: Automated Build (Recommended)

**Windows:**
```bash
.\build.bat
```

**Linux/Mac:**
```bash
bash build.sh
```

This will:
1. ✓ Validate Docker installation
2. ✓ Build all Docker images
3. ✓ Start 4 services (FastAPI, React, TimescaleDB, Redis)
4. ✓ Display service status

### Option 2: Manual Build

```bash
# Build images
docker-compose build --no-cache

# Start services
docker-compose up -d

# View logs
docker-compose logs -f ml_api
```

## Accessing the Application

Once services are running:

- **Frontend**: http://localhost:5173
  - Home page with project overview
  - Dashboard for video upload and processing

- **API Documentation**: http://localhost:8000/docs
  - Interactive Swagger UI
  - Try endpoints directly

- **Database**: Connect to `localhost:5432`
  - Username: `postgres`
  - Password: `password`
  - Database: `vidsum`

## Quick Test

1. Open http://localhost:5173 in your browser
2. Click "Try Dashboard" button
3. Select a video file (MP4, WebM, etc.)
4. Set target duration (10-300 seconds)
5. Choose selection method (Greedy or Knapsack)
6. Click "Upload & Process"
7. Watch real-time processing logs
8. Download summary when complete

## File Uploads

Videos are stored in:
- Uploads: `./data/uploads/`
- Outputs: `./data/outputs/`
- Temporary: `./data/temp/`

## Troubleshooting

### Services not starting
```bash
# Check service logs
docker-compose logs ml_api
docker-compose logs frontend
docker-compose logs db

# Restart all services
docker-compose restart
```

### Port conflicts
If ports 5173, 8000, 5432, or 6379 are in use:
1. Edit `docker-compose.yml`
2. Change port mappings
3. Rebuild: `docker-compose up -d --force-recreate`

### Out of memory
- Reduce `CHUNK_DURATION` in `vidsum_gnn/core/config.py` (default: 300s)
- Or increase Docker memory allocation

### GPU not detected
Ensure NVIDIA Docker runtime is installed:
```bash
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
```

If not working, remove `runtime: nvidia` from `docker-compose.yml`

## Development

### Backend Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn vidsum_gnn.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

## Stopping Services

```bash
# Stop all services
docker-compose down

# Remove volumes (careful - deletes data!)
docker-compose down -v

# View service status
docker-compose ps
```

## Architecture

```
┌─────────────────────────────────────────────┐
│         Frontend (React 18 + TypeScript)    │
│      Port 5173 - Vite Dev Server           │
│  ├─ HomePage: Project overview             │
│  └─ DashboardPage: Upload & monitoring     │
└────────────────┬────────────────────────────┘
                 │ HTTP/WebSocket
┌────────────────┴────────────────────────────┐
│         FastAPI Backend (uvicorn)           │
│      Port 8000 - REST API & WebSocket       │
│  ├─ /api/v1/videos/upload - File upload    │
│  ├─ /ws/logs/{video_id} - Real-time logs   │
│  └─ /docs - Swagger UI                     │
└────────────────┬────────────────────────────┘
                 │ asyncpg / redis
         ┌───────┴───────┐
         │               │
    ┌────▼──────┐  ┌────▼──────┐
    │ TimescaleDB│  │   Redis   │
    │ Port 5432 │  │ Port 6379 │
    └───────────┘  └───────────┘
```

## Performance Optimization

### For Long Videos (>1 hour)
1. Reduce chunk duration: `CHUNK_DURATION = 180` (3 min)
2. Enable GPU: Ensure NVIDIA Docker runtime
3. Monitor memory: Watch logs for memory alerts

### For Faster Processing
1. Reduce `GNN_NUM_HEADS` from 8 to 4 in config.py
2. Reduce `GNN_NUM_LAYERS` from 2 to 1
3. Use `--parallel` flag if implemented

## Data Cleanup

Remove old videos:
```bash
# List all videos in dashboard
# Delete via API (when implemented)

# Or manually
rm -rf data/uploads/* data/outputs/*
```

## Support & Issues

For issues:
1. Check logs: `docker-compose logs ml_api`
2. Review README.md for common issues
3. Validate video format: Use FFmpeg to test
   ```bash
   ffprobe your_video.mp4
   ```

## Next Steps

1. Train custom GNN model (optional)
2. Configure TimescaleDB compression (for production)
3. Set up monitoring dashboard
4. Deploy to cloud (AWS/GCP/Azure)

---

Happy summarizing! 🎬✨
