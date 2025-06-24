# TAI-Roaster Setup Guide

A comprehensive guide to set up and run the TAI-Roaster portfolio analysis system on your local machine.

## 🎯 Overview

TAI-Roaster is a portfolio analysis and recommendation system with:
- **Backend**: FastAPI-based API with AI/ML models
- **Frontend**: Next.js React application
- **Intelligence**: Advanced AI/ML models for portfolio analysis

## 📋 Prerequisites

### Required Software
1. **Python 3.12** (confirmed compatible)
2. **Node.js** (version 16 or higher)
3. **npm** (comes with Node.js)
4. **Git** (to clone the repository)

### System Requirements
- **Operating System**: macOS, Linux, or Windows
- **RAM**: Minimum 8GB (16GB recommended for ML models)
- **Storage**: At least 2GB free space

## 🚀 Installation Steps

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/[username]/TAI-Roaster.git

# Navigate to the project directory
cd TAI-Roaster
```

### Step 2: Backend Setup (Python)

#### Option A: Automated Setup (Recommended)
```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

#### Option B: Manual Setup
If the automated setup doesn't work:

```bash
# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
```

### Step 3: Frontend Setup (Node.js)

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Navigate back to root directory
cd ..
```

### Step 4: Environment Configuration (Optional)

The application can run without API keys, but for full functionality:

```bash
# Copy environment template
cp env_example.txt .env

# Edit the .env file with your API keys (optional)
nano .env  # or use your preferred editor
```

**Environment Variables:**
- `OPENAI_API_KEY`: For GPT-4 features (optional)
- `ANTHROPIC_API_KEY`: Alternative to OpenAI (optional)
- `DATABASE_URL`: Database connection (defaults to SQLite)
- `ENVIRONMENT`: Set to "development"
- `LOG_LEVEL`: Set to "INFO"

## 🏃 Running the Application

### Starting the Backend

**Option 1: From root directory**
```bash
# Navigate to root directory
cd TAI-Roaster

# Activate virtual environment
source venv/bin/activate

# Navigate to backend directory
cd backend

# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Option 2: One-liner command**
```bash
cd TAI-Roaster && source venv/bin/activate && cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at: `http://localhost:8000`

### Starting the Frontend

**In a new terminal:**
```bash
# Navigate to root directory
cd TAI-Roaster

# Navigate to frontend directory
cd frontend

# Start the Next.js development server
npm run dev
```

The frontend will be available at: `http://localhost:3000`

## 🔧 Verification

### Backend Health Check
Open your browser and visit:
- `http://localhost:8000` - Should show API information
- `http://localhost:8000/health` - Should return `{"status": "healthy"}`
- `http://localhost:8000/docs` - Interactive API documentation

### Frontend Check
Open your browser and visit:
- `http://localhost:3000` - Should show the TAI-Roaster web interface

## 📁 Project Structure

```
TAI-Roaster/
├── backend/           # FastAPI backend application
│   ├── app/          # Main application code
│   └── ...
├── frontend/          # Next.js frontend application
│   ├── app/          # Next.js 13+ app directory
│   ├── components/   # React components
│   └── ...
├── intelligence/      # AI/ML models and training
├── shared/           # Shared utilities
├── requirements.txt  # Python dependencies
└── setup.sh         # Automated setup script
```

## 🛠️ Development Workflow

### Daily Development
1. **Start Backend:**
   ```bash
   cd TAI-Roaster && source venv/bin/activate && cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start Frontend (new terminal):**
   ```bash
   cd TAI-Roaster/frontend && npm run dev
   ```

### Adding New Dependencies

**Python packages:**
```bash
# Activate virtual environment
source venv/bin/activate

# Install new package
pip install package-name

# Update requirements.txt
pip freeze > requirements.txt
```

**Node.js packages:**
```bash
# Navigate to frontend directory
cd frontend

# Install new package
npm install package-name
```

## �� Troubleshooting

### Common Issues

#### 1. Python Version Issues
```bash
# Check Python version
python3 --version
python3.12 --version

# If Python 3.12 not found, install it or use available Python 3.x
python3 -m venv venv
```

#### 2. Virtual Environment Issues
```bash
# Deactivate current environment
deactivate

# Remove existing venv
rm -rf venv

# Create new virtual environment
python3.12 -m venv venv
source venv/bin/activate
```

#### 3. Port Already in Use
```bash
# Kill process on port 8000
sudo lsof -t -i tcp:8000 | xargs kill -9

# Or use a different port
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

#### 4. Node.js Issues
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### 5. Permission Issues (macOS/Linux)
```bash
# Make setup script executable
chmod +x setup.sh

# Fix file permissions
chmod -R 755 TAI-Roaster/
```

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Ensure all prerequisites are installed
3. Verify that both backend and frontend are running
4. Check if ports 3000 and 8000 are available

## 📝 Additional Notes

### ML Models
- The application includes pre-trained ML models for portfolio analysis
- Models will load automatically when the backend starts
- Initial startup may take longer as models are loaded into memory

### Data Storage
- The application uses SQLite by default (no additional database setup required)
- Data files are stored in various directories (`data/`, `processed/`, `uploads/`)

### API Documentation
- Interactive API docs: `http://localhost:8000/docs`
- OpenAPI spec: `http://localhost:8000/openapi.json`

## ✅ Success Checklist

- [ ] Python 3.12 installed and working
- [ ] Node.js and npm installed
- [ ] Repository cloned successfully
- [ ] Virtual environment created and activated
- [ ] Python dependencies installed
- [ ] Frontend dependencies installed
- [ ] Backend starts without errors at `http://localhost:8000`
- [ ] Frontend starts without errors at `http://localhost:3000`
- [ ] Can access both applications in browser

## 🎉 Next Steps

Once everything is running:
1. Upload a portfolio CSV file through the web interface
2. Explore the analysis features
3. Check out the API documentation
4. Review the code structure for development

Happy coding! 🚀
