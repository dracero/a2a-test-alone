#!/usr/bin/env bash
# ==============================================================================
# Script: optimize_and_apply_harness.sh
# Propósito: Ejecutar la optimización recursiva (Propose -> Probe -> Compose) de
#            Harness² (Google Research) y aplicar la configuración ganadora a NAMS.
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
echo "🚀 [Harness² Offline] Optimización Recursiva y Aplicación en Caliente"
echo "======================================================================"
echo "📍 Raíz del proyecto: $ROOT_DIR"
echo "🐍 Intérprete Python: $PYTHON_BIN"
echo "📁 Dataset Benchmark: $ROOT_DIR/data/harness_benchmarks/student_queries_benchmark.json"
echo "⚙️ Configuración Destino: $ROOT_DIR/config/nams_harness_config.json"
echo "🔄 Modo: Propose -> Probe -> Compose (--probe-candidates --apply)"
echo "----------------------------------------------------------------------"

# Ejecutar ciclo completo recursivo de Harness² y persistir el ganador
$PYTHON_BIN "$ROOT_DIR/scripts/harness_offline_eval.py" --probe-candidates --apply "$@"

echo ""
echo "🎉 Optimización completada. La configuración ganadora ha sido guardada en:"
echo "   -> $ROOT_DIR/config/nams_harness_config.json"
echo "   NAMS y JEV consumirán estos parámetros calibrados de forma inmediata."

