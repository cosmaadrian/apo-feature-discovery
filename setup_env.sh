#!/bin/bash
# Script to create and activate Python virtual environment on Linux/macOS
# Usage: ./setup_env.sh

set -e  # Exit if any command fails


# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Make sure Python3 is installed."
    exit 1
fi

# Virtual environment name
VENV_NAME=".venv"

# Check if environment already exists
if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment already exists. Activating existing environment..."
    source "$VENV_NAME/bin/activate"
else
    # Create new virtual environment
    echo "Creating virtual environment '$VENV_NAME'..."
    python3 -m venv "$VENV_NAME"
    
    # Check if creation was successful
    if [ ! -f "$VENV_NAME/bin/activate" ]; then
        echo "ERROR: Virtual environment creation failed."
        exit 1
    fi
    
    # Activate environment
    echo "Activating virtual environment..."
    source "$VENV_NAME/bin/activate"

    # Update pip
    echo "Updating pip..."
    python -m pip install --upgrade pip

    # Install dependencies
    if [ -f "requirements.txt" ]; then
        echo "Installing dependencies from requirements.txt..."
        pip install -r requirements.txt
    else
        echo "WARNING: requirements.txt file not found."
    fi

    echo
    echo "========================================"
    echo "Virtual environment ready!"
    echo "========================================"
    echo
    echo "To activate the environment in the future use:"
    echo "  source $VENV_NAME/bin/activate"
    echo
    echo "To deactivate the environment use:"
    echo "  deactivate"
fi

