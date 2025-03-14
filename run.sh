#!/bin/bash

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Create directories if they don't exist
mkdir -p templates
mkdir -p static/css

# Run with gunicorn in production mode
if [ "$1" == "dev" ]; then
    echo "Starting in development mode..."
    python app.py
else
    echo "Starting in production mode with gunicorn..."
    gunicorn -w 4 -b 0.0.0.0:5000 app:app
fi 