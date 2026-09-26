"""
Script de auditoría de trazas de razonamiento y relaciones TOUCHED.

Permite inspeccionar:
- Acciones pedagógicas y modificaciones de ontología registradas.
- Modo de interacción (Profesor vs. Estudiante).
- Entidades del KG consultadas o alteradas (TOUCHED edges).
- Timestamp e insumos de decisión.

Uso:
    uv run python scripts/audit_reasoning_traces.py [--limit 20] [--agent physics]
"""

import os
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parents[1]
load_dotenv(root_dir / ".env", override=True)

import sys
sys.path.insert(0, str(root_dir / "demo/ui"))
from service.server.agent_nams import AgentNAMSMemory

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


async def main():
    parser = argparse.ArgumentParser(description="Auditoría de trazas de razonamiento NAMS")
    parser.add_argument("--limit", type=int, default=15, help="Número máximo de trazas a mostrar")
    parser.add_argument("--agent", type=str, default="physics", help="ID del agente a auditar")
    args = parser.parse_args()

    print("=" * 80)
    print(f"🔍 AUDITORÍA DE TRAZAS DE RAZONAMIENTO Y RELACIONES TOUCHED ({args.agent.upper()})")
    print("=" * 80)

    mem = AgentNAMSMemory(
        agent_id=args.agent,
        agent_name=f"Agente {args.agent}",
        uri=URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
    )

    if not await mem.ensure_connected():
        print("❌ No se pudo conectar a Neo4j.")
        return

    cypher = """
    MATCH (t:ReasoningTrace)
    WHERE t.agent_id = $agent_id OR $agent_id = 'all'
    OPTIONAL MATCH (t)-[r:TOUCHED]->(e:Entity)
    RETURN t.id AS trace_id,
           t.action AS action,
           t.mode AS mode,
           t.input_data AS input_data,
           t.output_data AS output_data,
           t.agent_id AS agent_id,
           t.is_correct AS is_correct,
           t.error_type AS error_type,
           t.severity AS severity,
           t.probability AS probability,
           toString(t.timestamp) AS timestamp,
           collect(CASE WHEN e IS NOT NULL THEN {
               id: e.id,
               name: e.name,
               type: e.type,
               action_type: r.action_type,
               mode: r.mode
           } ELSE null END) AS touched_entities
    ORDER BY timestamp DESC
    LIMIT $limit
    """

    results = await mem.client.long_term._client.execute_read(
        cypher,
        {"agent_id": args.agent, "limit": args.limit}
    )

    if not results:
        print("ℹ️ No se encontraron trazas de razonamiento para este agente.")
        return

    print(f"📊 Se encontraron {len(results)} traza(s) de razonamiento recientes:\n")

    for idx, trace in enumerate(results, 1):
        mode_str = f"[{trace.get('mode', 'N/A').upper()}]"
        print(f"{idx}. 🏷️  TRAZA: {trace.get('trace_id')} {mode_str}")
        print(f"    Acción:     {trace.get('action')}")
        print(f"    Fecha/Hora: {trace.get('timestamp')}")
        print(f"    Insumo:     {trace.get('input_data')}")
        print(f"    Decisión:   {trace.get('output_data')}")

        if trace.get("is_correct") is not None:
            correct_icon = "✅" if trace.get("is_correct") else "❌"
            err_type = trace.get("error_type") or "N/A"
            sev = trace.get("severity")
            prob = trace.get("probability")
            sev_str = f"{sev:.2f}/5.0" if isinstance(sev, (int, float)) else str(sev)
            prob_str = f"{prob:.2f}" if isinstance(prob, (int, float)) else str(prob)
            print(f"    Diagnóstico System One: {correct_icon} is_correct={trace.get('is_correct')} | Error={err_type} | Severidad={sev_str} | Prob={prob_str}")

        touched = [item for item in trace.get("touched_entities", []) if item is not None]
        if touched:
            print("    🔗 Entidades tocadas (TOUCHED):")
            for ent in touched:
                ent_name = ent.get("name") or ent.get("id")
                ent_type = ent.get("type", "Entity")
                print(f"       • {ent_name} ({ent_type})")
        else:
            print("    🔗 Entidades tocadas: (Ninguna)")
        print("-" * 80)

    print("\n✅ Auditoría finalizada.")


if __name__ == "__main__":
    asyncio.run(main())
