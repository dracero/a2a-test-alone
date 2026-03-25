#!/bin/bash

echo "Testing Medical Agent startup..."
echo "================================"

cd samples/python/agents/medical_Images

# Run the agent with timeout
timeout 5 uv run python -m app --host localhost --port 10002 2>&1 &
PID=$!

# Wait a bit for startup
sleep 3

# Kill the process
kill $PID 2>/dev/null

echo ""
echo "Test complete"
