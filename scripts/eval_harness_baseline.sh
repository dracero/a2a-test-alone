#!/usr/bin/env bash
# ==============================================================================
# Script: eval_harness_baseline.sh
# Propósito: Evaluar la configuración base del andamiaje NAMS mediante Harness²
#            (Google Research) contra el dataset golden de queries estudiantiles.
# ==============================================================================

set -euo pipefail

# Determinar directorio raíz del repositorio de forma dinámica
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Resolver intérprete de Python preferido
if [ -x "$ROOT_DIR/demo/ui/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/demo/ui/.venv/bin/python"
elif command -v uv &>/dev/null; then
    PYTHON_BIN="uv run python"
else
    PYTHON_BIN="python3"
fi

echo "======================================================================"
echo "🔬 [Harness² Offline] Evaluación de Configuración Base de Memoria NAMS"
echo "======================================================================"
echo "📍 Raíz del proyecto: $ROOT_DIR"
echo "🐍 Intérprete Python: $PYTHON_BIN"
echo "📁 Dataset Benchmark: $ROOT_DIR/data/harness_benchmarks/student_queries_benchmark.json"
echo "⚙️ Configuración NAMS: $ROOT_DIR/config/nams_harness_config.json"
echo "----------------------------------------------------------------------"

# Ejecutar evaluación de línea base
$PYTHON_BIN "$ROOT_DIR/scripts/harness_offline_eval.py" "$@"

echo ""
echo "✅ Evaluación de línea base completada con éxito."

