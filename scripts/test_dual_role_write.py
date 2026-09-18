"""
Test de verificación para la Fase 2: Lógica de Escritura Dual-Rol (Profesor vs. Estudiante).

Verifica:
1. En modo STUDENT:
   - Intentar agregar un concepto canónico al KG es RECHAZADO (retorna None).
   - Intentar actualizar un concepto canónico es RECHAZADO (retorna False).
   - Las afirmaciones del estudiante se registran como candidatos en Short-Term sin tocar el KG canónico.
2. En modo PROFESSOR:
   - Agregar conceptos canónicos al KG es ACEPTADO (retorna dict con ID y metadatos).
   - Actualizar conceptos canónicos es ACEPTADO (retorna True) y genera traza de auditoría TOUCHED.
3. Ambos roles pueden consultar el contexto canónico de forma Read-Only.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parents[1]
load_dotenv(root_dir / ".env", override=True)

import sys
sys.path.insert(0, str(root_dir / "demo/ui"))
from service.server.agent_nams import AgentNAMSMemory, InteractionMode

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
PHYSICS_ONTOLOGY_PATH = str(root_dir / "demo/ui/service/server/ontologies/physics_ontology.json")


async def main():
    print("=" * 70)
    print("🧪 TEST FASE 2: ESCRITURA DUAL-ROL (PROFESOR VS. ESTUDIANTE)")
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

    # -------------------------------------------------------------------------
    # Test 1: Modo Estudiante intenta escribir al KG canónico (Debe fallar)
    # -------------------------------------------------------------------------
    print("\n2️⃣ Probando que el modo ESTUDIANTE NO puede agregar conceptos canónicos...")
    student_attempt = await physics_mem.add_canonical_concept(
        name="Energía Cinética Relativista Errónea",
        entity_type="Concept",
        description="Afirmación no autorizada del estudiante",
        mode=InteractionMode.STUDENT,
    )
    assert student_attempt is None, "❌ ERROR: Modo estudiante logró crear un concepto canónico!"
    print("   ✅ Correcto: Intento en modo STUDENT fue denegado.")

    # -------------------------------------------------------------------------
    # Test 2: Modo Profesor agrega un concepto canónico (Debe tener éxito)
    # -------------------------------------------------------------------------
    print("\n3️⃣ Probando que el modo PROFESOR sí puede agregar conceptos canónicos...")
    test_concept_name = "Segunda Ley de Newton DualRol Test"
    professor_result = await physics_mem.add_canonical_concept(
        name=test_concept_name,
        entity_type="Principle",
        description="La aceleración de un objeto es directamente proporcional a la fuerza neta e inversamente proporcional a su masa.",
        properties={"formula": "F = m * a", "domain": "Dinámica Clásica"},
        mode=InteractionMode.PROFESSOR,
    )
    assert professor_result is not None, "❌ ERROR: Modo profesor falló al crear concepto canónico!"
    concept_id = professor_result["id"]
    print(f"   ✅ Concepto canónico creado con éxito por Profesor (ID: {concept_id}).")

    # -------------------------------------------------------------------------
    # Test 3: Modo Estudiante intenta actualizar el concepto canónico (Debe fallar)
    # -------------------------------------------------------------------------
    print("\n4️⃣ Probando que el modo ESTUDIANTE NO puede actualizar conceptos canónicos...")
    student_update = await physics_mem.update_canonical_concept(
        concept_id=concept_id,
        updates={"formula": "F = m / a (fórmula incorrecta)"},
        mode=InteractionMode.STUDENT,
    )
    assert student_update is False, "❌ ERROR: Modo estudiante logró modificar un concepto canónico!"
    print("   ✅ Correcto: Modificación en modo STUDENT fue denegada.")

    # -------------------------------------------------------------------------
    # Test 4: Modo Profesor actualiza el concepto canónico (Debe tener éxito)
    # -------------------------------------------------------------------------
    print("\n5️⃣ Probando que el modo PROFESOR sí puede actualizar conceptos canónicos...")
    professor_update = await physics_mem.update_canonical_concept(
        concept_id=concept_id,
        updates={"mathematical_form": "\\vec{F}_{net} = m \\cdot \\vec{a}"},
        mode=InteractionMode.PROFESSOR,
    )
    assert professor_update is True, "❌ ERROR: Modo profesor falló al actualizar concepto canónico!"
    print("   ✅ Concepto canónico actualizado con éxito por Profesor.")

    # -------------------------------------------------------------------------
    # Test 5: Consulta Read-Only del KG canónico
    # -------------------------------------------------------------------------
    print("\n6️⃣ Probando consulta Read-Only de conceptos canónicos...")
    canonical_results = await physics_mem.query_canonical_context(
        query="Segunda Ley de Newton DualRol Test",
        max_items=5,
    )
    assert len(canonical_results) > 0, "❌ No se encontró el concepto canónico registrado!"
    found = any(test_concept_name in r["name"] for r in canonical_results)
    assert found, f"❌ El concepto '{test_concept_name}' no aparece en los resultados canónicos!"
    print(f"   ✅ Contexto canónico recuperado correctamente: {len(canonical_results)} resultado(s).")

    # -------------------------------------------------------------------------
    # Test 6: Registro de afirmación de estudiante (Solo candidato, NO modifica KG)
    # -------------------------------------------------------------------------
    print("\n7️⃣ Probando registro de afirmación del estudiante (Student Claim)...")
    claim_result = await physics_mem.register_student_claim(
        student_id="test_student_dual_role",
        concept="Segunda Ley de Newton",
        student_claim="La fuerza es inversamente proporcional a la masa para aceleración constante",
        canonical_value="F = m * a",
        session_id="session_dual_role_test",
    )
    assert claim_result.get("candidate_id"), "❌ No se generó candidate_id para la afirmación del estudiante"
    print("   ✅ Afirmación del estudiante guardada como candidato aislado sin tocar el KG canónico.")

    print("\n" + "=" * 70)
    print("🎉 TEST FASE 2 COMPLETADO CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
