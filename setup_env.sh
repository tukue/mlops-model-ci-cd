#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting environment setup..."

# 1. Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    # Try 'python' first (common on Windows), then 'python3'
    if command -v python &> /dev/null; then
        python -m venv .venv
    else
        python3 -m venv .venv
    fi
else
    echo "Virtual environment (.venv) already exists."
fi

# 2. Activate Virtual Environment
# Detect OS/Shell to determine activation script
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash, etc.)
    source .venv/Scripts/activate
else
    # Linux/MacOS
    source .venv/bin/activate
fi

# 3. Install Dependencies
echo "Installing dependencies..."
# Use python -m pip to avoid file locking issues on Windows
python -m pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    python -m pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found."
fi

echo "----------------------------------------------------------------"
echo "Setup complete!"
echo ""
echo "To activate the environment in your current shell, run:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    echo "source .venv/Scripts/activate"
else
    echo "source .venv/bin/activate"
fi
echo "----------------------------------------------------------------"
