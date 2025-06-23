#!/bin/bash

# TA-Lib Installation Script for TAI-Roaster Intelligence Module
# This script installs TA-Lib library which is required for technical analysis

echo "🔧 Installing TA-Lib for TAI-Roaster Intelligence Module..."

# Check if TA-Lib is already installed
if python -c "import talib" 2>/dev/null; then
    echo "✅ TA-Lib is already installed"
    exit 0
fi

# Check operating system and install accordingly
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "📱 Detected macOS - Installing TA-Lib via Homebrew..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    # Install TA-Lib via Homebrew
    brew install ta-lib
    
    if [ $? -eq 0 ]; then
        echo "✅ TA-Lib system library installed successfully"
    else
        echo "❌ Failed to install TA-Lib system library"
        exit 1
    fi

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "🐧 Detected Linux - Installing TA-Lib from source..."
    
    # Create temporary directory
    temp_dir=$(mktemp -d)
    cd "$temp_dir"
    
    # Download TA-Lib source
    echo "📥 Downloading TA-Lib source..."
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to download TA-Lib source"
        cd - > /dev/null
        rm -rf "$temp_dir"
        exit 1
    fi
    
    # Extract and compile
    echo "📦 Extracting and compiling TA-Lib..."
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    
    ./configure --prefix=/usr
    make
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to compile TA-Lib"
        cd - > /dev/null
        rm -rf "$temp_dir"
        exit 1
    fi
    
    # Install (requires sudo)
    echo "🔐 Installing TA-Lib (requires sudo)..."
    sudo make install
    
    if [ $? -eq 0 ]; then
        echo "✅ TA-Lib system library installed successfully"
    else
        echo "❌ Failed to install TA-Lib system library"
        cd - > /dev/null
        rm -rf "$temp_dir"
        exit 1
    fi
    
    # Cleanup
    cd - > /dev/null
    rm -rf "$temp_dir"

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    echo "🪟 Detected Windows - Please install TA-Lib manually:"
    echo "   1. Download TA-Lib from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib"
    echo "   2. Install using: pip install TA_Lib-0.4.25-cp311-cp311-win_amd64.whl"
    echo "   3. Or use conda: conda install -c conda-forge ta-lib"
    exit 1

else
    echo "❌ Unsupported operating system: $OSTYPE"
    echo "Please install TA-Lib manually for your platform"
    exit 1
fi

# Install Python TA-Lib package
echo "🐍 Installing Python TA-Lib package..."
pip install TA-Lib

if [ $? -eq 0 ]; then
    echo "✅ Python TA-Lib package installed successfully"
else
    echo "❌ Failed to install Python TA-Lib package"
    echo "💡 You may need to:"
    echo "   - Ensure the system TA-Lib library is properly installed"
    echo "   - Update your library paths (ldconfig on Linux)"
    echo "   - Install development headers (apt-get install build-essential on Ubuntu)"
    exit 1
fi

# Verify installation
echo "🔍 Verifying TA-Lib installation..."
if python -c "import talib; print(f'✅ TA-Lib {talib.__version__} installed successfully')" 2>/dev/null; then
    echo "🎉 TA-Lib installation completed successfully!"
    echo ""
    echo "You can now use the intelligence module with technical analysis features."
else
    echo "❌ TA-Lib installation verification failed"
    echo "Please check the installation and try again"
    exit 1
fi 