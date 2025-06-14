#!/bin/bash

# Change to the current directory (optional, but safe)
cd "$(dirname "$0")" || exit 1

# Run Python HTTP server on port 8000
echo "Starting HTTP server on port 8000..."
python3 -m http.server 8000
