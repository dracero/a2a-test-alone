"""
Test de verificación para la Fase 3: Detección de Falencias por Acumulación Semántica.

Verifica:
1. Que registrar menos de 5 candidatos similares NO promueve a StudentDeficiency.
2. Que al acumular ≥5 afirmaciones erróneas semánticamente similares (cosine >= 0.82),
   el sistema las consolida automáticamente en una StudentDeficiency confirmada.
3. Que un candidato de un tema no relacionado (ej: óptica) NO se agrupa con el cluster de dinámica.
4. Que la falencia confirmada queda vinculada al perfil del estudiante y al concepto en Neo4j.
"""

import os
import asyncio
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
PHYSICS_ONTOLOGY_PATH = str(root_dir / "demo/ui/service/server/ontologies/physics_ontology.json")
STUDENT_ID = "student_semantic_test_01"


async def cleanup_test_data(mem: AgentNAMSMemory, student_id: str):
    """Limpia los datos de prueba anteriores para este estudiante."""
    if not await mem.ensure_connected():
        return
    cypher = """
    MATCH (c:MisconceptionCandidate {student_id: $student_id})
    DETACH DELETE c
    WITH 1 AS dummy
    MATCH (d:StudentDeficiency {student_id: $student_id})
    DETACH DELETE d
    """
    try:
        await mem.client.long_term._client.execute_write(cypher, {"student_id": student_id})
    except Exception as e:
        print(f"Nota: Limpieza previa: {e}")


async def main():
    print("=" * 70)
    print("🧪 TEST FASE 3: DETECCIÓN SEMÁNTICA DE FALENCIAS (UMBRAL ≥ 5)")
    print("=" * 70)

    physics_mem = AgentNAMSMemory(
        agent_id="physics",
        agent_name="Tutor Socrático de Física Multimodal",
        uri=URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        ontology_path=PHYSICS_ONTOLOGY_PATH,
    )

    print("\n1️⃣ Conectando a Neo4j...")
    connected = await physics_mem.ensure_connected()
    assert connected, "❌ Falló conexión a Neo4j"
    print("   ✅ Conectado a Neo4j.")

    # Limpiar datos previos
    await cleanup_test_data(physics_mem, STUDENT_ID)

    # -------------------------------------------------------------------------
    # Test 1: Registrar 4 candidatos similares (Debajo del umbral de 5)
    # -------------------------------------------------------------------------
    print("\n2️⃣ Registrando 4 afirmaciones erróneas similares sobre 'Fuerza de Rozamiento'...")
    friction_claims = [
        "La fuerza de rozamiento estático siempre es igual a mu estático por la normal.",
        "El rozamiento en reposo se calcula directamente como el coeficiente mu multiplicado por N.",
        "Para un bloque que no se mueve en un plano inclinado, la fricción siempre vale mu por la normal.",
        "Siempre que hay fricción estática la fórmula a usar es F_roce = mu_s * Normal.",
    ]

    for idx, claim in enumerate(friction_claims, 1):
        res = await physics_mem.register_misconception_candidate(
            student_id=STUDENT_ID,
            concept="Fuerza de Rozamiento Estático",
            student_claim=claim,
            canonical_value="La fuerza de rozamiento estática se calcula mediante las ecuaciones de equilibrio (sumatoria de fuerzas = 0). La fórmula mu*N solo determina su valor MÁXIMO posible.",
            session_id=f"sess_friction_{idx}",
        )
        assert res.get("candidate_id"), f"❌ Falló registro de candidato {idx}"
        print(f"   📝 Candidato {idx} registrado: '{claim[:50]}...'")

    print("\n3️⃣ Intentando consolidar con 4 candidatos (Umbral requerido: 5)...")
    consolidated_4 = await physics_mem.consolidate_deficiencies(
        student_id=STUDENT_ID,
        similarity_threshold=0.82,
        count_threshold=5,
    )
    assert len(consolidated_4) == 0, f"❌ ERROR: Se consolidó con {len(friction_claims)} candidatos! Deberían ser 0."
    print("   ✅ Correcto: Con 4 candidatos NO se consolidó ninguna falencia (umbral ≥ 5 respetado).")

    # -------------------------------------------------------------------------
    # Test 2: Registrar 5to candidato similar + 1 candidato no relacionado (Óptica)
    # -------------------------------------------------------------------------
    print("\n4️⃣ Registrando el 5to candidato similar sobre 'Fuerza de Rozamiento'...")
    claim_5 = "La fuerza de roce estática en reposo se obtiene multiplicando el coeficiente de fricción por la normal."
    await physics_mem.register_misconception_candidate(
        student_id=STUDENT_ID,
        concept="Fuerza de Rozamiento Estático",
        student_claim=claim_5,
        canonical_value="F_roce estática surge del equilibrio dinámico/estático; mu*N es solo la cota máxima.",
        session_id="sess_friction_5",
    )
    print(f"   📝 Candidato 5 registrado: '{claim_5[:50]}...'")

    print("\n5️⃣ Registrando un candidato sobre un tema COMPLETAMENTE DISTINTO (Óptica)...")
    optics_claim = "En refracción de la luz el índice n hace que la velocidad sea infinita en el vacío."
    await physics_mem.register_misconception_candidate(
        student_id=STUDENT_ID,
        concept="Refracción y Ley de Snell",
        student_claim=optics_claim,
        canonical_value="El índice de refracción n = c/v, donde c es la velocidad de la luz en el vacío.",
        session_id="sess_optics_1",
    )
    print(f"   📝 Candidato no relacionado registrado: '{optics_claim[:50]}...'")

    # -------------------------------------------------------------------------
    # Test 3: Ejecutar consolidación semántica (Debe confirmar 1 falencia de 5 miembros)
    # -------------------------------------------------------------------------
    print("\n6️⃣ Ejecutando consolidación semántica con 6 candidatos totales...")
    consolidated = await physics_mem.consolidate_deficiencies(
        student_id=STUDENT_ID,
        similarity_threshold=0.82,
        count_threshold=5,
    )

    assert len(consolidated) == 1, f"❌ ERROR: Se esperaban 1 falencia consolidada, se obtuvieron {len(consolidated)}!"
    deficiency = consolidated[0]
    print(f"   🎯 ¡Falencia confirmada exitosamente!")
    print(f"      - Concepto: {deficiency['concept']}")
    print(f"      - Ocurrencias agrupadas: {deficiency['occurrences']}")
    print(f"      - Explicación: {deficiency['description']}")

    assert deficiency["occurrences"] == 5, f"❌ Se debieron agrupar exactamente los 5 candidatos de fricción, agrupó: {deficiency['occurrences']}"
    print("   ✅ Verificación de clustering: El candidato de Óptica quedó correctamente EXCLUIDO.")

    # -------------------------------------------------------------------------
    # Test 4: Consultar las falencias confirmadas del estudiante en Neo4j
    # -------------------------------------------------------------------------
    print("\n7️⃣ Consultando falencias confirmadas del estudiante en el KG...")
    student_defs = await physics_mem.get_student_deficiencies(STUDENT_ID)
    assert len(student_defs) >= 1, "❌ No se encontraron falencias confirmadas en el grafo!"
    print(f"   ✅ Se recuperaron {len(student_defs)} falencia(s) confirmada(s) para '{STUDENT_ID}'.")
    print(f"      - Nodo ID: {student_defs[0]['id']}")
    print(f"      - Concepto: {student_defs[0]['concept']}")
    print(f"      - Severidad: {student_defs[0]['severity']}")
    print(f"      - Ocurrencias: {student_defs[0]['occurrences']}")

    print("\n" + "=" * 70)
    print("🎉 TEST FASE 3 COMPLETADO CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
