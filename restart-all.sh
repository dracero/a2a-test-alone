#!/bin/bash

# Script para reiniciar todos los servicios del sistema A2A

echo "🛑 Deteniendo todos los servicios..."

# Detener procesos Python (backend y agentes)
pkill -f "uv run python" 2>/dev/null
pkill -f "python -m app" 2>/dev/null
pkill -f "python main.py" 2>/dev/null

# Detener frontend
pkill -f "npm run dev" 2>/dev/null
pkill -f "next dev" 2>/dev/null

sleep 2

echo "✅ Servicios detenidos"
echo ""
echo "🚀 Iniciando servicios..."
echo ""

# Iniciar backend
echo "📦 Iniciando backend orchestrator..."
cd demo/ui
uv run python main.py > ../../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"
cd ../..

sleep 3

# Iniciar agentes
echo "🤖 Iniciando agentes..."

echo "   - Medical Images Agent..."
cd samples/python/agents/medical_Images
uv run python -m app > ../../../../logs/medical.log 2>&1 &
MEDICAL_PID=$!
echo "     PID: $MEDICAL_PID"
cd ../../../..

echo "   - Multimodal Agent..."
cd samples/python/agents/multimodal
uv run python -m app > ../../../../logs/multimodal.log 2>&1 &
MULTIMODAL_PID=$!
echo "     PID: $MULTIMODAL_PID"
cd ../../../..

echo "   - Images Agent..."
cd samples/python/agents/images
uv run python -m app > ../../../../logs/images.log 2>&1 &
IMAGES_PID=$!
echo "     PID: $IMAGES_PID"
cd ../../../..

sleep 2

# Iniciar frontend
echo "🎨 Iniciando frontend..."
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"
cd ..

echo ""
echo "✅ Todos los servicios iniciados!"
echo ""
echo "📋 PIDs de los procesos:"
echo "   Backend:    $BACKEND_PID"
echo "   Medical:    $MEDICAL_PID"
echo "   Multimodal: $MULTIMODAL_PID"
echo "   Images:     $IMAGES_PID"
echo "   Frontend:   $FRONTEND_PID"
echo ""
echo "📊 Para ver los logs:"
echo "   tail -f logs/backend.log"
echo "   tail -f logs/medical.log"
echo "   tail -f logs/multimodal.log"
echo "   tail -f logs/images.log"
echo "   tail -f logs/frontend.log"
echo ""
echo "🌐 URLs:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:12000"
echo ""
