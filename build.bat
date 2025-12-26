@echo off
REM Build script for VIDSUM-GNN project (Windows)

echo ===== VIDSUM-GNN Build Script =====
echo.

REM Check Docker
echo [1/5] Checking Docker installation...
docker --version >nul 2>&1
if errorlevel 1 (
    echo X Docker not found. Please install Docker Desktop.
    exit /b 1
)
echo OK Docker found

REM Check Docker Compose
echo [2/5] Checking Docker Compose...
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo X Docker Compose not found.
    exit /b 1
)
echo OK Docker Compose found

REM Build images
echo [3/5] Building Docker images...
docker-compose build --no-cache
if errorlevel 1 (
    echo X Build failed
    exit /b 1
)

REM Start services
echo [4/5] Starting services...
docker-compose up -d
if errorlevel 1 (
    echo X Failed to start services
    exit /b 1
)

REM Wait for services
echo [5/5] Waiting for services to be ready...
timeout /t 10 /nobreak

REM Check services
echo.
echo ===== Service Status =====
docker-compose ps

echo.
echo OK Build complete!
echo.
echo Access the application:
echo   Frontend: http://localhost:5173
echo   API Docs: http://localhost:8000/docs
echo.
echo View logs with: docker-compose logs -f
echo Stop services with: docker-compose down
