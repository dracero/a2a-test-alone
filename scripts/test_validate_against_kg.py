"""
Test de verificación para la Fase 4: Validación contra Contenido Canónico del KG.

Verifica:
1. Que un concepto canónico registrado en modo Profesor sea consultable.
2. Que una afirmación errónea del estudiante que contradiga el KG sea detectada como is_correct: False
   y su corrección canónica no sea aceptada como verdad.
3. Que una afirmación correcta del estudiante alineada con el KG sea validada como is_correct: True.
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
    print("🧪 TEST FASE 4: VALIDACIÓN CONTRA CONTENIDO CANÓNICO DEL KG")
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

    # 1. Asegurar concepto canónico registrado por el Profesor
    print("\n2️⃣ Registrando concepto canónico 'Tercera Ley de Newton' en modo Profesor...")
    await physics_mem.add_canonical_concept(
        name="Tercera Ley de Newton - Acción y Reacción",
        entity_type="Principle",
        description="A toda acción se opone una reacción igual y contraria. Las fuerzas de acción y reacción actúan siempre sobre CUERPOS DISTINTOS y por lo tanto nunca se anulan mutuamente en el diagrama de cuerpo libre de un único objeto.",
        properties={
            "formula": "\\vec{F}_{A\\to B} = -\\vec{F}_{B\\to A}",
            "domain": "Dinámica Clásica",
        },
        mode=InteractionMode.PROFESSOR,
    )
    print("   ✅ Concepto canónico registrado en el KG.")

    # 2. Test: Afirmación incorrecta del estudiante
    print("\n3️⃣ Probando validación de afirmación INCORRECTA del estudiante...")
    incorrect_claim = "Las fuerzas de acción y reacción se anulan mutuamente porque tienen igual magnitud y sentido contrario sobre el mismo cuerpo."
    print(f"   Afirmación: \"{incorrect_claim}\"")

    val_incorrect = await physics_mem.validate_against_kg(student_claim=incorrect_claim, student_id="test_student_1")
    print(f"   Resultado de validación híbrida (System One + System Two):")
    print(f"      - is_correct:      {val_incorrect.get('is_correct')}")
    print(f"      - probability:     {val_incorrect.get('probability')}")
    print(f"      - error_type:      {val_incorrect.get('error_type')}")
    print(f"      - severity:        {val_incorrect.get('severity')}/5.0")
    print(f"      - concept:         {val_incorrect.get('concept')}")
    print(f"      - canonical_value: {val_incorrect.get('canonical_value')}")
    print(f"      - explanation:     {val_incorrect.get('explanation')}")
    print(f"      - socratic_reply:  {val_incorrect.get('socratic_reply')}")
    print(f"      - trace_id:        {val_incorrect.get('trace_id')}")

    assert val_incorrect.get("is_correct") is False, "❌ ERROR: La afirmación errónea fue aceptada como correcta!"
    valid_error_types = [
        "CONTRADICCION_DE_PRINCIPIO_RECTOR",
        "ATRIBUCION_ERRONEA_DE_PROPIEDADES",
        "CONFUSION_CONCEPTUAL_O_ESTRUCTURAL",
        "APLICACION_FUERA_DE_DOMINIO_O_CONDICION",
    ]
    assert val_incorrect.get("error_type") in valid_error_types, f"❌ ERROR: Tipo de error inesperado: {val_incorrect.get('error_type')}"
    assert float(val_incorrect.get("severity", 0)) >= 2.5, f"❌ ERROR: Severidad esperada >= 2.5, obtenida: {val_incorrect.get('severity')}"
    assert bool(val_incorrect.get("socratic_reply")), "❌ ERROR: Falta socratic_reply de Gemini 3.5 Flash!"
    print(f"   ✅ Correcto: Afirmación incorrecta RECHAZADA con diagnóstico universal ({val_incorrect.get('error_type')}) y réplica socrática generada.")

    # Verificar traza y arista TOUCHED en Neo4j
    trace_id = val_incorrect.get("trace_id")
    if trace_id:
        cypher_check = """
        MATCH (t:ReasoningTrace {id: $trace_id})-[r:TOUCHED]->(e:Entity)
        RETURN t.is_correct AS is_correct, t.error_type AS error_type, t.severity AS severity, e.name AS entity_name
        """
        rows = await physics_mem.client.long_term._client.execute_read(cypher_check, {"trace_id": trace_id})
        print(f"   🔍 Verificación en Neo4j para traza {trace_id}: {len(rows)} conexión(es) TOUCHED encontradas.")
        for r in rows:
            print(f"      -> [:TOUCHED] con '{r.get('entity_name')}' | error_type={r.get('error_type')}, sev={r.get('severity')}")

    # 3. Test: Afirmación correcta del estudiante
    print("\n4️⃣ Probando validación de afirmación CORRECTA del estudiante (Física)...")
    correct_claim = "Las fuerzas del par acción y reacción nunca pueden anularse entre sí porque actúan sobre cuerpos diferentes."
    print(f"   Afirmación: \"{correct_claim}\"")

    val_correct = await physics_mem.validate_against_kg(student_claim=correct_claim, student_id="test_student_1")
    print(f"   Resultado de validación híbrida:")
    print(f"      - is_correct:      {val_correct.get('is_correct')}")
    print(f"      - probability:     {val_correct.get('probability')}")
    print(f"      - error_type:      {val_correct.get('error_type')}")
    print(f"      - severity:        {val_correct.get('severity')}/5.0")
    print(f"      - explanation:     {val_correct.get('explanation')}")
    print(f"      - socratic_reply:  {val_correct.get('socratic_reply')}")
    print(f"      - trace_id:        {val_correct.get('trace_id')}")

    assert val_correct.get("is_correct") is True, "❌ ERROR: La afirmación correcta fue rechazada!"
    assert val_correct.get("error_type") == "NINGUNO", f"❌ ERROR: Error inesperado en afirmación válida: {val_correct.get('error_type')}"
    assert float(val_correct.get("severity", 5.0)) <= 1.5, f"❌ ERROR: Severidad esperada <= 1.5, obtenida: {val_correct.get('severity')}"
    print("   ✅ Correcto: La afirmación correcta fue ACEPTADA con tipado exacto.")

    # 4. Test Multi-Dominio: Medicina / Histología (Taxonomía Universal sin sesgo de Física)
    print("\n5️⃣ Probando validación en dominio MÉDICO (Histología) con Taxonomía Universal...")
    medical_mem = AgentNAMSMemory(
        agent_id="medical",
        agent_name="Agente Médico Histopatológico",
        uri=URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
    )
    await medical_mem.ensure_connected()

    # Registrar concepto canónico médico
    await medical_mem.add_canonical_concept(
        name="Tejido Epitelial - Características y Nutrición",
        entity_type="Concept",
        description="El tejido epitelial es completamente avascular, carece de vasos sanguíneos propios y se nutre por difusión a través de la membrana basal desde el tejido conectivo adyacente.",
        mode=InteractionMode.PROFESSOR,
    )

    medical_incorrect = "El tejido epitelial posee una rica red vascular de capilares sanguíneos que irrigan directamente entre las células epiteliales."
    print(f"   Afirmación médica errónea: \"{medical_incorrect}\"")

    val_med = await medical_mem.validate_against_kg(student_claim=medical_incorrect, student_id="med_student_1")
    print(f"   Resultado validación Médica (Taxonomía Universal):")
    print(f"      - is_correct:      {val_med.get('is_correct')}")
    print(f"      - probability:     {val_med.get('probability')}")
    print(f"      - error_type:      {val_med.get('error_type')}")
    print(f"      - severity:        {val_med.get('severity')}/5.0")
    print(f"      - socratic_reply:  {val_med.get('socratic_reply')}")

    assert val_med.get("is_correct") is False, "❌ ERROR: Afirmación médica errónea aceptada!"
    assert val_med.get("error_type") in ["ATRIBUCION_ERRONEA_DE_PROPIEDADES", "CONFUSION_CONCEPTUAL_O_ESTRUCTURAL", "CONTRADICCION_DE_PRINCIPIO_RECTOR"], f"Error inesperado: {val_med.get('error_type')}"
    print(f"   ✅ Correcto: Error médico diagnosticado como '{val_med.get('error_type')}' de forma 100% genérica.")

    print("\n" + "=" * 70)
    print("🎉 TEST MULTI-DOMINIO UNIVERSAL (FÍSICA + MEDICINA) COMPLETADO CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
