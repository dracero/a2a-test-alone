#!/bin/bash

echo "🔧 Instalando dependencias de los agentes..."
echo ""

# Multimodal Agent (Port 10003)
echo "📦 Instalando dependencias del agente multimodal (puerto 10003)..."
cd samples/python/agents/multimodal
uv sync
if [ $? -eq 0 ]; then
    echo "✅ Agente multimodal instalado correctamente"
else
    echo "❌ Error instalando agente multimodal"
    exit 1
fi
cd ../../../..

echo ""
echo "✅ Agente instalado correctamente"
echo ""
echo "Para iniciar todos los servicios, ejecuta:"
echo "  npm run dev:all"
