"""
Script batch de consolidación de falencias de estudiantes por acumulación semántica.

Recorre los agentes activos y estudiantes registrados, analizando candidatos
a misconception con embeddings semánticos.
Promueve a StudentDeficiency confirmada únicamente aquellos clusters que superen
el umbral de 5 coincidencias con similitud coseno >= 0.82.

Uso:
    uv run python scripts/consolidate_student_deficiencies.py
"""

import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parents[1]
load_dotenv(root_dir / ".env", override=True)

import sys
sys.path.insert(0, str(root_dir / "demo/ui"))
from service.server.agent_nams import AgentNAMSMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
PHYSICS_ONTOLOGY_PATH = str(root_dir / "demo/ui/service/server/ontologies/physics_ontology.json")


async def get_active_students_with_pending_candidates(mem: AgentNAMSMemory) -> list[str]:
    """Consulta la lista de student_ids que tienen candidatos a falencia pendientes."""
    if not await mem.ensure_connected():
        return []

    try:
        cypher = """
        MATCH (c:MisconceptionCandidate {agent_id: $agent_id, status: 'pending'})
        RETURN DISTINCT c.student_id AS student_id
        """
        results = await mem.client.long_term._client.execute_read(
            cypher,
            {"agent_id": mem.agent_id}
        )
        return [row["student_id"] for row in results if row.get("student_id")]
    except Exception as e:
        logger.error(f"Error consultando estudiantes con candidatos pendientes: {e}")
        return []


async def main():
    print("=" * 70)
    print("🧠 CONSOLIDACIÓN BATCH DE FALENCIAS POR SIMILARIDAD SEMÁNTICA")
    print(f"   Umbral de coincidencia: {AgentNAMSMemory.DEFICIENCY_COUNT_THRESHOLD} o más")
    print(f"   Similitud coseno mínima: {AgentNAMSMemory.DEFICIENCY_SIMILARITY_THRESHOLD}")
    print("=" * 70)

    # Memorias de agentes
    physics_mem = AgentNAMSMemory(
        agent_id="physics",
        agent_name="Tutor Socrático de Física Multimodal",
        uri=URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        ontology_path=PHYSICS_ONTOLOGY_PATH,
    )
    medical_mem = AgentNAMSMemory(
        agent_id="medical",
        agent_name="Asistente Médico",
        uri=URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
    )

    agents = [physics_mem, medical_mem]
    total_consolidated = 0

    for mem in agents:
        connected = await mem.ensure_connected()
        if not connected:
            logger.warning(f"⚠️ No se pudo conectar a la memoria de {mem.agent_name}. Saltando...")
            continue

        students = await get_active_students_with_pending_candidates(mem)
        print(f"\n📚 [{mem.agent_name}] Estudiantes con candidatos pendientes: {len(students)}")

        for student_id in students:
            new_deficiencies = await mem.consolidate_deficiencies(
                student_id=student_id,
                similarity_threshold=AgentNAMSMemory.DEFICIENCY_SIMILARITY_THRESHOLD,
                count_threshold=AgentNAMSMemory.DEFICIENCY_COUNT_THRESHOLD,
            )
            if new_deficiencies:
                total_consolidated += len(new_deficiencies)
                print(f"   🎯 [{student_id}] {len(new_deficiencies)} nueva(s) falencia(s) confirmada(s):")
                for d in new_deficiencies:
                    print(f"      - Concepto: '{d['concept']}' ({d['occurrences']} coincidencias)")

    print("\n" + "=" * 70)
    print(f"🏁 Consolidación finalizada. Total de nuevas falencias confirmadas: {total_consolidated}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
