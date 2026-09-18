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

    val_incorrect = await physics_mem.validate_against_kg(student_claim=incorrect_claim)
    print(f"   Resultado de validación:")
    print(f"      - is_correct: {val_incorrect.get('is_correct')}")
    print(f"      - concept: {val_incorrect.get('concept')}")
    print(f"      - canonical_value: {val_incorrect.get('canonical_value')}")
    print(f"      - explanation: {val_incorrect.get('explanation')}")

    assert val_incorrect.get("is_correct") is False, "❌ ERROR: La afirmación errónea fue aceptada como correcta!"
    print("   ✅ Correcto: La afirmación incorrecta fue RECHAZADA (is_correct: False).")

    # 3. Test: Afirmación correcta del estudiante
    print("\n4️⃣ Probando validación de afirmación CORRECTA del estudiante...")
    correct_claim = "Las fuerzas del par acción y reacción nunca pueden anularse entre sí porque actúan sobre cuerpos diferentes."
    print(f"   Afirmación: \"{correct_claim}\"")

    val_correct = await physics_mem.validate_against_kg(student_claim=correct_claim)
    print(f"   Resultado de validación:")
    print(f"      - is_correct: {val_correct.get('is_correct')}")
    print(f"      - explanation: {val_correct.get('explanation')}")

    assert val_correct.get("is_correct") is True, "❌ ERROR: La afirmación correcta fue rechazada!"
    print("   ✅ Correcto: La afirmación correcta fue ACEPTADA (is_correct: True).")

    print("\n" + "=" * 70)
    print("🎉 TEST FASE 4 COMPLETADO CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
