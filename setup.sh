#!/bin/bash

echo "🚀 Setting up TAI-Roaster Development Environment"
echo "================================================="

# Check if Python 3.13 is available
if command -v python3.13 &> /dev/null; then
    PYTHON_CMD="python3.13"
    echo "✅ Found Python 3.13"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✅ Found Python 3"
else
    echo "❌ Python 3 not found. Please install Python 3.13 or later."
    exit 1
fi

echo "Using Python command: $PYTHON_CMD"
$PYTHON_CMD --version

# Create virtual environment
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start the backend server:"
echo "  cd backend"
echo "  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "To start the frontend (in a new terminal):"
echo "  cd frontend"
echo "  npm install"
echo "  npm run dev" 