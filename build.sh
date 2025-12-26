#!/bin/bash
# Build script for VIDSUM-GNN project

set -e

echo "===== VIDSUM-GNN Build Script ====="
echo ""

# Check Docker
echo "[1/5] Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop."
    exit 1
fi
echo "✓ Docker found"

# Check Docker Compose
echo "[2/5] Checking Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose."
    exit 1
fi
echo "✓ Docker Compose found"

# Build images
echo "[3/5] Building Docker images..."
docker-compose build --no-cache

# Start services
echo "[4/5] Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "[5/5] Waiting for services to be ready..."
sleep 10

# Check services
echo ""
echo "===== Service Status ====="
docker-compose ps

echo ""
echo "✓ Build complete!"
echo ""
echo "Access the application:"
echo "  Frontend: http://localhost:5173"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "View logs with: docker-compose logs -f"
echo "Stop services with: docker-compose down"
