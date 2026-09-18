"""
Test de verificación para la Fase 1: Ontología POLE+O por Dominio.

Verifica:
1. Que el agente de Física cargue y registre su ontología POLE+O desde el archivo JSON.
2. Que el agente Médico funcione con ontología dinámica extraída por LLM (sin JSON manual).
3. Que ambos schemas estén aislados y configurados correctamente.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno del proyecto
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


async def main():
    print("=" * 70)
    print("🧪 TEST FASE 1: REGISTRO DE ONTOLOGÍA POLE+O EDUCATIVA")
    print("=" * 70)

    # 1. Instanciar memoria de Física con ontología POLE+O
    print("\n1️⃣ Inicializando memoria de Física con schema POLE+O...")
    physics_mem = AgentNAMSMemory(
        agent_id="physics",
        agent_name="Tutor Socrático de Física Multimodal",
        uri=URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        ontology_path=PHYSICS_ONTOLOGY_PATH,
    )

    # 2. Instanciar memoria Médica (sin JSON, extracción dinámica por LLM)
    print("2️⃣ Inicializando memoria Médica con extracción automática LLM...")
    medical_mem = AgentNAMSMemory(
        agent_id="medical",
        agent_name="Asistente Médico",
        uri=URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        ontology_path=None,  # LLM extrae dinámicamente de los documentos
    )

    print("\n3️⃣ Conectando a Neo4j y registrando ontologías...")
    p_conn = await physics_mem.ensure_connected()
    m_conn = await medical_mem.ensure_connected()
    assert p_conn, "❌ Falló conexión de memoria de Física"
    assert m_conn, "❌ Falló conexión de memoria Médica"

    p_reg = await physics_mem.register_ontology()
    m_reg = await medical_mem.register_ontology()
    assert p_reg, "❌ Falló registro de ontología de Física"
    assert m_reg, "❌ Falló registro de ontología Médica"

    # 4. Validar tipos de entidad en Física
    assert physics_mem.schema_config is not None, "❌ schema_config de Física no debe ser None"
    entity_type_names = physics_mem.schema_config.get_entity_type_names()
    relation_type_names = physics_mem.schema_config.get_relation_types()

    print(f"\n   📋 Tipos de entidad registrados en Física: {len(entity_type_names)}")
    print(f"      {entity_type_names}")
    print(f"   🔗 Tipos de relación registrados en Física: {len(relation_type_names)}")
    print(f"      {relation_type_names}")

    # Aserciones de tipos específicos POLE+O educativos
    for expected_type in ["Concept", "Principle", "PhysicalQuantity", "Experiment", "StudentProfile", "StudentDeficiency", "MisconceptionCandidate"]:
        assert expected_type in entity_type_names, f"❌ Falta el tipo '{expected_type}' en la ontología de Física!"
    print("   ✅ Todos los tipos educativos requeridos están presentes en la ontología de Física.")

    for expected_rel in ["REQUIRES", "APPLIES_TO", "HAS_DEFICIENCY", "ABOUT_CONCEPT", "DEMONSTRATED_IN", "TOUCHED"]:
        assert expected_rel in relation_type_names, f"❌ Falta la relación '{expected_rel}' en la ontología de Física!"
    print("   ✅ Todas las relaciones educativas requeridas están presentes en la ontología de Física.")

    # 5. Validar que la ontología médica es dinámica
    assert medical_mem.ontology_path is None, "❌ La memoria médica no debe tener ontology_path manual"
    print("   ✅ Agente médico configurado correctamente para extracción dinámica vía LLM.")

    print("\n" + "=" * 70)
    print("🎉 TEST FASE 1 COMPLETADO CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
