#!/bin/bash
# Quick Setup Script for VideoSum-GNN Model Integration
# Run this script from the project root directory

set -e  # Exit on error

echo "========================================="
echo "VideoSum-GNN Model Integration Setup"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Check Python
echo -e "${YELLOW}[1/7] Checking Python installation...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}Python not found. Please install Python 3.10+${NC}"
    exit 1
fi
PYTHON_VERSION=$(python --version)
echo -e "${GREEN}✓ Found $PYTHON_VERSION${NC}"
echo ""

# 2. Check Node.js
echo -e "${YELLOW}[2/7] Checking Node.js installation...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js not found. Please install Node.js 18+${NC}"
    exit 1
fi
NODE_VERSION=$(node --version)
echo -e "${GREEN}✓ Found Node.js $NODE_VERSION${NC}"
echo ""

# 3. Check PostgreSQL
echo -e "${YELLOW}[3/7] Checking PostgreSQL...${NC}"
if ! command -v psql &> /dev/null; then
    echo -e "${RED}PostgreSQL not found. Please install PostgreSQL 14+${NC}"
    echo "Or use Docker: docker run -d --name vidsum-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=vidsum_gnn -p 5432:5432 postgres:14-alpine"
    exit 1
fi
echo -e "${GREEN}✓ PostgreSQL found${NC}"
echo ""

# 4. Create and setup database
echo -e "${YELLOW}[4/7] Setting up database...${NC}"
read -p "Enter PostgreSQL username (default: postgres): " DB_USER
DB_USER=${DB_USER:-postgres}

read -p "Create vidsum_gnn database? (y/n): " CREATE_DB
if [ "$CREATE_DB" = "y" ]; then
    psql -U $DB_USER -c "DROP DATABASE IF EXISTS vidsum_gnn;" || echo "Database doesn't exist yet"
    psql -U $DB_USER -c "CREATE DATABASE vidsum_gnn;"
    echo -e "${GREEN}✓ Database created${NC}"
else
    echo -e "${YELLOW}⚠ Skipping database creation${NC}"
fi
echo ""

# 5. Setup Python environment
echo -e "${YELLOW}[5/7] Setting up Python environment...${NC}"
if [ ! -d "venv" ]; then
    python -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

source venv/bin/activate || source venv/Scripts/activate  # Windows compatibility
echo -e "${GREEN}✓ Virtual environment activated${NC}"

echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements-local.txt
pip install -e .
echo -e "${GREEN}✓ Python packages installed${NC}"
echo ""

# 6. Initialize database schema
echo -e "${YELLOW}[6/7] Initializing database schema...${NC}"
python -c "
from vidsum_gnn.db.models import Base
from vidsum_gnn.db.client import engine
import asyncio

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print('✓ Database schema created')
    
asyncio.run(init_db())
"

echo "Applying database optimizations..."
python optimize_database.py
echo -e "${GREEN}✓ Database initialized${NC}"
echo ""

# 7. Setup frontend
echo -e "${YELLOW}[7/7] Setting up frontend...${NC}"
cd frontend
echo "Installing npm packages..."
npm install
npm install @radix-ui/react-tabs
echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
cd ..
echo ""

# 8. Verify model checkpoint
echo -e "${YELLOW}Verifying model checkpoint...${NC}"
if [ -f "model/models/checkpoints/best_model.pt" ]; then
    echo -e "${GREEN}✓ Model checkpoint found${NC}"
else
    echo -e "${RED}⚠ Model checkpoint not found at model/models/checkpoints/best_model.pt${NC}"
    echo "  Please train the model first or copy the checkpoint to this location"
fi
echo ""

# 9. Download pre-trained models (optional)
read -p "Pre-download AI models (Whisper, Flan-T5, Sentence-BERT)? This will download ~1.4GB (y/n): " DOWNLOAD_MODELS
if [ "$DOWNLOAD_MODELS" = "y" ]; then
    echo -e "${YELLOW}Downloading models... (this may take a few minutes)${NC}"
    python -c "
from transformers import WhisperProcessor, WhisperForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

print('Downloading Whisper base...')
WhisperProcessor.from_pretrained('openai/whisper-base')
WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')

print('Downloading Sentence-BERT...')
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print('Downloading Flan-T5 base...')
T5Tokenizer.from_pretrained('google/flan-t5-base')
T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')

print('✓ All models downloaded')
    "
    echo -e "${GREEN}✓ AI models downloaded${NC}"
else
    echo -e "${YELLOW}⚠ Skipping model download. Models will be downloaded on first use.${NC}"
fi
echo ""

# 10. Create startup scripts
echo -e "${YELLOW}Creating startup scripts...${NC}"

# Backend startup script
cat > start_backend.sh << 'EOF'
#!/bin/bash
source venv/bin/activate || source venv/Scripts/activate
cd vidsum_gnn
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
EOF
chmod +x start_backend.sh

# Frontend startup script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
cd frontend
npm run dev
EOF
chmod +x start_frontend.sh

echo -e "${GREEN}✓ Startup scripts created${NC}"
echo ""

# Summary
echo "========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start the backend:"
echo "   ./start_backend.sh"
echo "   (or manually: cd vidsum_gnn && uvicorn api.main:app --reload)"
echo ""
echo "2. In a new terminal, start the frontend:"
echo "   ./start_frontend.sh"
echo "   (or manually: cd frontend && npm run dev)"
echo ""
echo "3. Open your browser:"
echo "   Frontend: http://localhost:5173"
echo "   Backend API docs: http://localhost:8000/docs"
echo ""
echo "4. Upload a test video and select summary type!"
echo ""
echo "Documentation:"
echo "  - INTEGRATION_GUIDE.md - Complete setup guide"
echo "  - API_DOCUMENTATION.md - API reference"
echo "  - DEPLOYMENT_CHECKLIST.md - Testing guide"
echo "  - INTEGRATION_SUMMARY.md - Quick overview"
echo ""
echo "Troubleshooting:"
echo "  - Check database connection in vidsum_gnn/core/config.py"
echo "  - Verify model checkpoint exists: model/models/checkpoints/best_model.pt"
echo "  - See DEPLOYMENT_CHECKLIST.md for common issues"
echo ""
echo -e "${GREEN}Happy summarizing! 🎬${NC}"
