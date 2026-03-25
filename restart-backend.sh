#!/bin/bash

echo "🔄 Reiniciando el backend..."

# Buscar y matar el proceso del backend
pkill -f "uv run main.py" || echo "No se encontró proceso del backend corriendo"

# Esperar un momento
sleep 2

# Reiniciar el backend
echo "🚀 Iniciando backend..."
cd demo/ui && uv run main.py &

echo "✅ Backend reiniciado en el puerto 12000"
echo "Presiona Ctrl+C para detener"

# Mantener el script corriendo
wait
