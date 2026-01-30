#!/bin/bash

# Autonomous Credentialing System - Monolith Setup Script
# This script sets up both frontend and backend

set -e  # Exit on error

echo "=================================================="
echo "  Autonomous Credentialing System - Full Stack"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}✗ Node.js is not installed. Please install Node.js 18+${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js found: $(node --version)${NC}"

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}✗ npm is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ npm found: $(npm --version)${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed. Please install Python 3.9+${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python found: $(python3 --version)${NC}"

echo ""
echo "=================================================="
echo "  Setting up Backend (Python)"
echo "=================================================="
echo ""

cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating backend .env file..."
    cp .env.template .env
    echo -e "${YELLOW}! Please edit backend/.env with your API keys${NC}"
else
    echo -e "${GREEN}✓ Backend .env file exists${NC}"
fi

# Create necessary directories
mkdir -p logs exports
echo -e "${GREEN}✓ Backend directories created${NC}"

cd ..

echo ""
echo "=================================================="
echo "  Setting up Frontend (Next.js)"
echo "=================================================="
echo ""

cd frontend

# Install Node dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
    echo -e "${GREEN}✓ Node.js dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Node.js dependencies already installed${NC}"
fi

# Create .env.local if it doesn't exist
if [ ! -f ".env.local" ]; then
    echo "Creating frontend .env.local file..."
    cat > .env.local << EOF
NEXT_PUBLIC_API_URL=http://localhost:5000/api
NEXT_PUBLIC_WS_URL=ws://localhost:5000
EOF
    echo -e "${GREEN}✓ Frontend .env.local created${NC}"
else
    echo -e "${GREEN}✓ Frontend .env.local exists${NC}"
fi

cd ..

echo ""
echo "=================================================="
echo "  Database Setup"
echo "=================================================="
echo ""

echo "Database setup instructions:"
echo "1. Go to https://supabase.com"
echo "2. Create a new project"
echo "3. Go to SQL Editor"
echo "4. Copy and paste the contents of database/supabase_schema.sql"
echo "5. Execute the SQL"
echo "6. Get your database credentials from Settings > Database"
echo "7. Update backend/.env with Supabase credentials"
echo ""
echo -e "${YELLOW}Press Enter when database is set up...${NC}"
read

echo ""
echo "=================================================="
echo "  Testing Connections"
echo "=================================================="
echo ""

# Test backend setup
echo "Testing backend setup..."
cd backend
source venv/bin/activate

# Create test script
cat > test_setup.py << 'EOF'
import os
from dotenv import load_dotenv

load_dotenv()

checks = []

# Check environment variables
print("Checking environment variables...")
required_vars = [
    'LANGSMITH_API_KEY',
    'OPENAI_API_KEY',
    'TWILIO_ACCOUNT_SID',
    'DEEPGRAM_API_KEY',
    'ELEVENLABS_API_KEY',
    'SUPABASE_PASSWORD'
]

for var in required_vars:
    if os.getenv(var):
        print(f"✓ {var} is set")
        checks.append(True)
    else:
        print(f"✗ {var} is missing")
        checks.append(False)

if all(checks):
    print("\n✓ All required environment variables are set!")
else:
    print("\n✗ Some environment variables are missing. Please update .env")

# Test database connection
try:
    from credentialing_agent import DatabaseManager
    db = DatabaseManager()
    print("\n✓ Database connection successful!")
    db.close()
except Exception as e:
    print(f"\n✗ Database connection failed: {e}")
    print("  Please check your Supabase credentials")
EOF

python test_setup.py
rm test_setup.py

cd ..

echo ""
echo "=================================================="
echo "  Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start the backend:"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   python api_server.py"
echo ""
echo "2. In another terminal, start the frontend:"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo "3. Open your browser:"
echo "   http://localhost:3000"
echo ""
echo "4. Configure Twilio webhooks:"
echo "   - Voice URL: https://your-domain.com/webhook/voice"
echo "   - Status Callback: https://your-domain.com/webhook/status"
echo ""
echo "For production deployment:"
echo "   docker-compose -f docker-compose-monolith.yml up -d"
echo ""
echo "=================================================="
echo ""
