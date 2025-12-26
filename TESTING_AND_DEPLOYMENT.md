# VIDSUM-GNN Complete System Testing & Deployment Guide

## Quick Status Check

### ✅ Project Completion Summary
- **Backend**: FastAPI microservice with 8 REST endpoints + WebSocket support
- **Frontend**: React 18 with TypeScript, Vite, TailwindCSS, shadcn/ui components
- **Database**: TimescaleDB with optimizations (indexes, hypertables, compression)
- **Processing**: Batch-wise video processing with memory management
- **Logging**: Real-time structured logging with WebSocket streaming
- **Infrastructure**: Docker containerization (4 services: ml_api, frontend, db, redis)

### 📋 Completed Components

#### Backend (vidsum_gnn/)
- ✅ FastAPI application with lifespan management
- ✅ WebSocket endpoint for real-time logs (/ws/logs/{video_id})
- ✅ 8 REST endpoints (upload, process, status, results, etc.)
- ✅ Structured logging with StructuredLogger class
- ✅ PipelineStage enum for tracking stages
- ✅ Memory management (clear_memory, gc.collect, cuda cache clearing)
- ✅ Batch processing with 300s chunks and 30s overlap
- ✅ FFmpeg integration for video processing
- ✅ Feature extraction (ViT-B/16 visual, Wav2Vec2 audio)
- ✅ Graph construction with PyTorch Geometric
- ✅ GNN inference with Graph Attention Networks

#### Frontend (frontend/)
- ✅ React 18 + TypeScript
- ✅ Vite for fast development
- ✅ TailwindCSS for styling
- ✅ shadcn/ui components (Button, Card, Input, Progress, Slider)
- ✅ HomePage with project showcase
- ✅ DashboardPage with upload, controls, logs, progress
- ✅ React Router v6 for navigation
- ✅ Axios for API calls
- ✅ WebSocket client for real-time logs
- ✅ File drag-and-drop support
- ✅ Target duration slider (10-300s)
- ✅ Selection method radio buttons (greedy/knapsack)

#### Database (TimescaleDB)
- ✅ Hypertables for shots, embeddings, summaries
- ✅ Compression policies for older data
- ✅ Multiple indexes for query optimization
- ✅ Continuous aggregates for fast analytics
- ✅ Retention policies (optional)
- ✅ Database statistics views

#### Infrastructure
- ✅ Docker Compose with 4 services
- ✅ Redis for caching
- ✅ TimescaleDB with PostgreSQL
- ✅ Environment variables configuration
- ✅ Volume mounts for persistence

---

## Pre-Deployment Checklist

### System Requirements
- [ ] Docker & Docker Compose installed
- [ ] Python 3.9+ (for local development)
- [ ] FFmpeg installed
- [ ] GPU with CUDA 12.1 (optional, CPU works too)
- [ ] 8GB+ RAM
- [ ] 20GB+ free disk space

### Port Availability
- [ ] Port 5173 available (Frontend)
- [ ] Port 8000 available (API)
- [ ] Port 5432 available (Database)
- [ ] Port 6379 available (Redis)

---

## Quick Start (Docker)

### 1. Start All Services
```bash
# From project root
docker-compose up -d

# Verify services
docker-compose ps
```

Expected output:
```
NAME                    STATUS          PORTS
vidsum_gnn_frontend    Up X seconds    0.0.0.0:5173->5173/tcp
vidsum_gnn_ml_api      Up X seconds    0.0.0.0:8000->8000/tcp
vidsum_gnn_db          Up X seconds    0.0.0.0:5432->5432/tcp
vidsum_gnn_redis       Up X seconds    0.0.0.0:6379->6379/tcp
```

### 2. Access the Application
- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

### 3. Stop Services
```bash
docker-compose down

# To remove volumes and start fresh
docker-compose down -v
```

---

## Testing Procedures

### Test 1: Frontend Loading
```
Goal: Verify frontend loads without errors
Steps:
1. Open http://localhost:5173 in browser
2. Verify HomePage displays:
   - VIDSUM-GNN title
   - Processing Pipeline section
   - Features cards
   - Technology Stack
   - Team section
3. Click "Try Dashboard" link
4. Verify DashboardPage loads with:
   - Upload section
   - Target Duration slider
   - Selection Method radio buttons
   - Status card
   - Processing Logs section
```

### Test 2: API Health Check
```
Goal: Verify API is responding
Steps:
1. curl http://localhost:8000/health
2. Expected response:
   {
     "status": "healthy",
     "timestamp": "ISO-8601-timestamp"
   }

Or use browser:
http://localhost:8000/docs
- View all available endpoints
- Test endpoints interactively
```

### Test 3: File Upload Test
```
Goal: Test video upload functionality
Steps:
1. Open http://localhost:5173/dashboard
2. Prepare a test video (5-10 minutes recommended)
3. Select video in upload area (drag or click)
4. Configure settings:
   - Target Duration: 30s
   - Selection Method: greedy
5. Click "Upload & Process"

Expected behavior:
- Progress bar starts moving
- Logs appear in real-time
- Status changes to "processing"
- WebSocket connects successfully
```

### Test 4: Real-time Logs Streaming
```
Goal: Verify WebSocket logging works
Steps:
1. Upload a video (see Test 3)
2. Watch Processing Logs section
3. Verify logs appear in real-time with:
   - Timestamps
   - Log levels (INFO, ERROR, WARNING, SUCCESS)
   - Stage information
   - Progress percentages

Expected pipeline stages:
1. UPLOAD - File received
2. SHOT_DETECTION - Finding key scenes
3. FEATURE_EXTRACTION - Extracting visual/audio features
4. GRAPH_CONSTRUCTION - Building scene graph
5. GNN_INFERENCE - Calculating importance scores
6. SUMMARY_ASSEMBLY - Creating final video
7. COMPLETED - Success
```

### Test 5: Processing Completion
```
Goal: Verify full processing pipeline
Steps:
1. Complete Test 3 (upload and process)
2. Wait for processing to complete (varies by video length)
3. Verify status changes to "completed"
4. Verify summary preview appears
5. Click "Download Summary" button
6. Verify video file downloads successfully

Processing time estimates:
- Upload: < 1 minute
- Shot Detection: 1-2 minutes (depends on video)
- Feature Extraction: 2-3 minutes
- Graph Construction: 1 minute
- GNN Inference: 1-2 minutes
- Summary Assembly: < 1 minute
Total: 6-10 minutes for 10-minute video
```

### Test 6: Database Connectivity
```
Goal: Verify database is working
Steps:
1. Connect with psql:
   psql -h localhost -U postgres -d vidsum_gnn_db -W
   Password: password

2. Run queries:
   SELECT COUNT(*) FROM videos;
   SELECT COUNT(*) FROM shots;
   SELECT COUNT(*) FROM embeddings;

3. Verify hypertables exist:
   SELECT * FROM timescaledb_information.hypertable;
```

### Test 7: Redis Cache
```
Goal: Verify Redis is operational
Steps:
1. Open another terminal
2. Connect to Redis:
   docker-compose exec redis redis-cli

3. Test commands:
   PING
   SET test "hello"
   GET test
   DEL test
```

### Test 8: Memory Management
```
Goal: Verify GPU/CPU memory is being managed
Steps:
1. Monitor during processing:
   docker stats vidsum_gnn_ml_api

2. Verify memory usage:
   - Doesn't continuously grow
   - Returns to baseline after batch completion
   - No OOM (Out of Memory) errors

3. Check logs for memory alerts:
   docker logs vidsum_gnn_ml_api | grep -i memory
```

### Test 9: Error Handling
```
Goal: Test system resilience
Steps:

A. Invalid File Format:
1. Try uploading non-video file
2. Expect error message in logs
3. Status should show "error"

B. Oversized File:
1. Try uploading very large file (>5GB)
2. Should handle gracefully

C. Missing Configuration:
1. Verify .env or config settings
2. System should use defaults or show clear errors

D. Network Issues:
1. Close browser tab during processing
2. Stop Docker service and restart
3. Verify resumption or error handling
```

### Test 10: Performance Metrics
```
Goal: Verify system performance
Steps:
1. Upload multiple videos in sequence
2. Monitor metrics:
   - API response time: < 1s
   - WebSocket latency: < 100ms
   - Database query time: < 500ms
   - Frontend load time: < 2s

Commands:
   docker stats  # Real-time resource usage
   docker logs -f vidsum_gnn_ml_api  # API logs with timestamps
```

---

## Troubleshooting

### Issue: Frontend not loading
```
Solution:
1. docker-compose logs vidsum_gnn_frontend
2. Check for npm errors
3. Verify node_modules installed: docker-compose exec frontend npm ls
4. Rebuild: docker-compose down && docker-compose up -d --build frontend
```

### Issue: API returning 500 errors
```
Solution:
1. docker logs vidsum_gnn_ml_api
2. Check Python syntax: docker-compose exec ml_api python -m py_compile /app/vidsum_gnn/api/main.py
3. Verify database connection: docker-compose logs vidsum_gnn_db
4. Check TimescaleDB: docker-compose exec db psql -U postgres -c "SELECT 1"
```

### Issue: Uploads failing
```
Solution:
1. Check upload directory exists: docker-compose exec ml_api ls -la /app/data/uploads/
2. Verify permissions: docker exec vidsum_gnn_ml_api chmod 777 /app/data/
3. Check FFmpeg: docker-compose exec ml_api ffmpeg -version
4. Check disk space: docker exec vidsum_gnn_ml_api df -h
```

### Issue: WebSocket not connecting
```
Solution:
1. Verify ConnectionManager in main.py: docker logs vidsum_gnn_ml_api | grep -i websocket
2. Check CORS settings: http://localhost:8000/docs
3. Browser console: Press F12 and check Network tab
4. Restart service: docker-compose restart ml_api
```

### Issue: Database connection errors
```
Solution:
1. Check if DB service is running: docker-compose ps
2. Verify credentials in config: cat vidsum_gnn/core/config.py
3. Test connection: docker-compose exec db psql -U postgres -c "SELECT version();"
4. Check logs: docker-compose logs vidsum_gnn_db
```

### Issue: GPU not being used
```
Solution:
1. Verify CUDA available: docker exec vidsum_gnn_ml_api python -c "import torch; print(torch.cuda.is_available())"
2. Check GPU drivers: nvidia-smi
3. Update docker-compose.yml with proper GPU runtime
4. Falls back to CPU automatically if GPU unavailable
```

---

## Production Deployment Checklist

### Before Going Live
- [ ] Change default passwords (database, Redis)
- [ ] Update environment variables (.env file)
- [ ] Set LOG_LEVEL to "WARNING" instead of "DEBUG"
- [ ] Enable HTTPS/SSL certificates
- [ ] Set up proper backup strategy
- [ ] Configure log rotation
- [ ] Set retention policies for old data
- [ ] Test auto-scaling (if on cloud platform)
- [ ] Set up monitoring and alerting
- [ ] Configure firewall rules

### Security Hardening
```bash
# 1. Generate secure passwords
openssl rand -base64 32

# 2. Update environment variables
# In .env file:
DB_PASSWORD=<generated-password>
REDIS_PASSWORD=<generated-password>
API_KEY=<generated-key>

# 3. Update docker-compose.yml with secrets
# See Docker secrets documentation

# 4. Enable SSL in reverse proxy
# Use Nginx, Traefik, or cloud provider's load balancer
```

### Backup Strategy
```bash
# Backup database
docker-compose exec db pg_dump -U postgres vidsum_gnn_db > backup.sql

# Restore from backup
docker-compose exec db psql -U postgres vidsum_gnn_db < backup.sql

# Backup volumes
docker run --rm -v vidsum_gnn_data:/data -v $(pwd):/backup \
  busybox tar czf /backup/data-backup.tar.gz -C / data
```

### Monitoring Setup
```bash
# View real-time logs
docker-compose logs -f

# Monitor resource usage
docker stats

# Get system info
docker-compose exec ml_api python -c "
import psutil
import torch
print(f'CPU: {psutil.cpu_percent()}%')
print(f'RAM: {psutil.virtual_memory().percent}%')
print(f'GPU: {torch.cuda.is_available()}')
"
```

---

## Performance Optimization

### Database Optimization
```sql
-- Run these in psql
VACUUM ANALYZE videos;
VACUUM ANALYZE shots;
VACUUM ANALYZE embeddings;
VACUUM ANALYZE summaries;

-- Enable query statistics
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- View slow queries
SELECT query, calls, mean_time FROM pg_stat_statements
WHERE mean_time > 100
ORDER BY mean_time DESC;
```

### API Optimization
```python
# In vidsum_gnn/core/config.py
UVICORN_WORKERS = 4  # Increase for multi-core systems
LOG_LEVEL = "WARNING"  # Reduce logging verbosity
BATCH_SIZE = 16  # Adjust based on GPU memory
```

### Frontend Optimization
```bash
# Build optimized bundle
docker-compose exec frontend npm run build

# Check bundle size
docker-compose exec frontend npm run build -- --analyze
```

---

## Support & Debugging

### Collect Diagnostic Information
```bash
#!/bin/bash
# Save this as diagnose.sh

echo "=== System Information ===" > diagnosis.txt
docker-compose ps >> diagnosis.txt
docker stats --no-stream >> diagnosis.txt

echo "\n=== Docker Logs ===" >> diagnosis.txt
docker-compose logs --tail=50 >> diagnosis.txt

echo "\n=== API Health ===" >> diagnosis.txt
curl -s http://localhost:8000/health >> diagnosis.txt

echo "\n=== Database Status ===" >> diagnosis.txt
docker-compose exec db psql -U postgres -c "SELECT datname, pg_size_pretty(pg_database_size(datname)) FROM pg_database" >> diagnosis.txt

echo "Diagnostic file saved to: diagnosis.txt"
```

### Contact Information
For issues:
1. Check logs: `docker-compose logs <service-name>`
2. Check documentation in README.md
3. Review this testing guide
4. Check GitHub issues if applicable

---

## Next Steps

After successful testing:
1. ✅ Document any custom configurations
2. ✅ Set up automated backups
3. ✅ Configure monitoring and alerts
4. ✅ Create runbooks for operations
5. ✅ Train team on system operation
6. ✅ Set up CI/CD pipeline for future updates

---

## Appendix: Common Commands

```bash
# View logs
docker-compose logs -f ml_api
docker-compose logs -f frontend

# Execute commands in container
docker-compose exec ml_api python -c "import torch; print(torch.cuda.is_available())"

# Interactive shell
docker-compose exec ml_api /bin/bash
docker-compose exec frontend /bin/sh

# Rebuild specific service
docker-compose up -d --build ml_api

# Full restart
docker-compose down
docker-compose up -d

# Check health
docker-compose ps
curl http://localhost:8000/health

# Database access
docker-compose exec db psql -U postgres -d vidsum_gnn_db

# Clean up
docker system prune -a  # Remove all unused images/containers
docker volume prune  # Remove unused volumes
```

---

**Last Updated**: 2024-12-25  
**Status**: Production Ready ✅  
**Version**: 1.0.0
