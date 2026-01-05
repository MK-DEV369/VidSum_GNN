@echo off
REM Quick Setup Script for VideoSum-GNN Model Integration (Windows)
REM Run this script from the project root directory

setlocal enabledelayedexpansion

echo =========================================
echo VideoSum-GNN Model Integration Setup
echo =========================================
echo.

REM 1. Check Python
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo OK Found %PYTHON_VERSION%
echo.

REM 2. Check Node.js
echo [2/7] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found. Please install Node.js 18+
    exit /b 1
)
for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
echo OK Found Node.js %NODE_VERSION%
echo.

REM 3. Check PostgreSQL
echo [3/7] Checking PostgreSQL...
psql --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: PostgreSQL not found. Please install PostgreSQL 14+
    echo Or use Docker: docker run -d --name vidsum-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=vidsum_gnn -p 5432:5432 postgres:14-alpine
    exit /b 1
)
echo OK PostgreSQL found
echo.

REM 4. Create and setup database
echo [4/7] Setting up database...
set /p DB_USER="Enter PostgreSQL username (default: postgres): "
if "%DB_USER%"=="" set DB_USER=postgres

set /p CREATE_DB="Create vidsum_gnn database? (y/n): "
if /i "%CREATE_DB%"=="y" (
    psql -U %DB_USER% -c "DROP DATABASE IF EXISTS vidsum_gnn;" 2>nul
    psql -U %DB_USER% -c "CREATE DATABASE vidsum_gnn;"
    if errorlevel 1 (
        echo WARNING: Database creation failed. Please create manually.
    ) else (
        echo OK Database created
    )
) else (
    echo WARNING: Skipping database creation
)
echo.

REM 5. Setup Python environment
echo [5/7] Setting up Python environment...
if not exist "venv" (
    python -m venv venv
    echo OK Virtual environment created
)

call venv\Scripts\activate.bat
echo OK Virtual environment activated

echo Installing Python packages...
python -m pip install --upgrade pip
pip install -r requirements-local.txt
pip install -e .
echo OK Python packages installed
echo.

REM 6. Initialize database schema
echo [6/7] Initializing database schema...
python -c "from vidsum_gnn.db.models import Base; from vidsum_gnn.db.client import engine; import asyncio; async def init_db(): async with engine.begin() as conn: await conn.run_sync(Base.metadata.create_all); print('OK Database schema created'); asyncio.run(init_db())"

echo Applying database optimizations...
python optimize_database.py
echo OK Database initialized
echo.

REM 7. Setup frontend
echo [7/7] Setting up frontend...
cd frontend
echo Installing npm packages...
call npm install
call npm install @radix-ui/react-tabs
echo OK Frontend dependencies installed
cd ..
echo.

REM 8. Verify model checkpoint
echo Verifying model checkpoint...
if exist "model\models\checkpoints\best_model.pt" (
    echo OK Model checkpoint found
) else (
    echo WARNING: Model checkpoint not found at model\models\checkpoints\best_model.pt
    echo   Please train the model first or copy the checkpoint to this location
)
echo.

REM 9. Download pre-trained models (optional)
set /p DOWNLOAD_MODELS="Pre-download AI models (Whisper, Flan-T5, Sentence-BERT)? This will download ~1.4GB (y/n): "
if /i "%DOWNLOAD_MODELS%"=="y" (
    echo Downloading models... ^(this may take a few minutes^)
    python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration; from sentence_transformers import SentenceTransformer; print('Downloading Whisper base...'); WhisperProcessor.from_pretrained('openai/whisper-base'); WhisperForConditionalGeneration.from_pretrained('openai/whisper-base'); print('Downloading Sentence-BERT...'); SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('Downloading Flan-T5 base...'); T5Tokenizer.from_pretrained('google/flan-t5-base'); T5ForConditionalGeneration.from_pretrained('google/flan-t5-base'); print('OK All models downloaded')"
    echo OK AI models downloaded
) else (
    echo WARNING: Skipping model download. Models will be downloaded on first use.
)
echo.

REM 10. Create startup scripts
echo Creating startup scripts...

REM Backend startup script
echo @echo off > start_backend.bat
echo call venv\Scripts\activate.bat >> start_backend.bat
echo cd vidsum_gnn >> start_backend.bat
echo uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 >> start_backend.bat

REM Frontend startup script
echo @echo off > start_frontend.bat
echo cd frontend >> start_frontend.bat
echo npm run dev >> start_frontend.bat

echo OK Startup scripts created
echo.

REM Summary
echo =========================================
echo Setup Complete!
echo =========================================
echo.
echo Next steps:
echo.
echo 1. Start the backend:
echo    start_backend.bat
echo    ^(or manually: cd vidsum_gnn ^&^& uvicorn api.main:app --reload^)
echo.
echo 2. In a new terminal, start the frontend:
echo    start_frontend.bat
echo    ^(or manually: cd frontend ^&^& npm run dev^)
echo.
echo 3. Open your browser:
echo    Frontend: http://localhost:5173
echo    Backend API docs: http://localhost:8000/docs
echo.
echo 4. Upload a test video and select summary type!
echo.
echo Documentation:
echo   - INTEGRATION_GUIDE.md - Complete setup guide
echo   - API_DOCUMENTATION.md - API reference
echo   - DEPLOYMENT_CHECKLIST.md - Testing guide
echo   - INTEGRATION_SUMMARY.md - Quick overview
echo.
echo Troubleshooting:
echo   - Check database connection in vidsum_gnn\core\config.py
echo   - Verify model checkpoint exists: model\models\checkpoints\best_model.pt
echo   - See DEPLOYMENT_CHECKLIST.md for common issues
echo.
echo Happy summarizing! 🎬
echo.

pause
