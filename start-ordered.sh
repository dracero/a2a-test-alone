#!/bin/bash

export NODE_OPTIONS="--no-deprecation"

# Function to wait for a port to be ready
wait_for_port() {
  local port=$1
  local name=$2
  echo "--------------------------------------------------"
  echo "⏳ Waiting for $name on port $port..."
  while ! nc -z localhost $port; do
    sleep 2
  done
  echo "✅ $name is ready!"
}

# Trap SIGINT (Ctrl+C) and SIGTERM to kill background processes
cleanup() {
  echo ""
  echo "🛑 Stopping all services..."
  # Kill all child processes of this script
  pkill -P $$
  exit
}
trap cleanup SIGINT SIGTERM

echo "🚀 Starting BeeAI Ecosystem in order..."

# 0. Ensure Local Databases (Neo4j & Qdrant)
if ! nc -z localhost 7687 2>/dev/null; then
  echo "🧠 Starting local Neo4j container..."
  docker start neo4j-local >/dev/null 2>&1 || true
  wait_for_port 7687 "Neo4j Local (NAMS)"
else
  echo "✅ Neo4j Local (NAMS) is active on port 7687."
fi

if ! nc -z localhost 6333 2>/dev/null; then
  echo "📦 Starting local Qdrant container..."
  docker start qdrant-local-histo >/dev/null 2>&1 || true
  wait_for_port 6333 "Qdrant Vector DB"
else
  echo "✅ Qdrant Local is active on port 6333."
fi

# 1. Start Priority Agent (Multimodal)
echo "Step 1: Starting Priority Agent (10003)..."
npm run dev:agent:multimodal &

# Wait for priority agent
wait_for_port 10003 "Multimodal Agent"

# 2. Start Remaining Agents (Images & Medical)
echo ""
echo "Step 2: Starting Remaining Agents (10001 & 10002)..."
npm run dev:agent:images &
npm run dev:agent:medical &

# Wait for remaining agents
wait_for_port 10001 "Images Agent"
wait_for_port 10002 "Medical Agent"

# 3. Start Orchestrator
echo ""
echo "Step 3: Starting Orchestrator (Backend)..."
npm run dev:backend &
wait_for_port 12000 "Orchestrator"

# 4. Start Frontend
echo ""
echo "Step 4: Starting Frontend..."
echo "--------------------------------------------------"
echo "Frontend will be available at http://localhost:3000"
echo "Press Ctrl+C to stop everything."
echo "--------------------------------------------------"
npm run dev:frontend
