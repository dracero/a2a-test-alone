"""
Script de utilidad para etiquetar y aislar nodos existentes en Neo4j Aura.
Asigna `agent_id` a conversaciones y entidades existentes para evitar
que entidades de Histología aparezcan en consultas de Física (y viceversa).
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

# Cargar variables de entorno del proyecto
root_dir = Path(__file__).resolve().parents[1]
load_dotenv(root_dir / ".env", override=True)

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


async def tag_existing_nodes():
    if not (URI and USERNAME and PASSWORD):
        print("❌ Credenciales de Neo4j no encontradas en .env")
        return

    print(f"🔗 Conectando a Neo4j Aura: {URI} (db: {DATABASE})...")
    async with AsyncGraphDatabase.driver(URI, auth=(USERNAME, PASSWORD)) as driver:
        async with driver.session() as session:
            # 1. Tagging de conversaciones por contenido de mensajes
            print("\n📌 Etiquetando conversaciones existentes según su contenido...")
            tag_medical_convs = """
            MATCH (c:Conversation)-[:CONTAINS]->(m:Message)
            WHERE (m.content =~ '(?i).*(histolog|arteria|célula|celula|tejido|endotelio|epitelio|médic|medic|túnica|tunica|órgano|organo|microscop).*')
              AND (c.agent_id IS NULL OR NOT c.session_id STARTS WITH 'medical:')
            SET c.agent_id = 'medical',
                c.agent_name = 'Asistente Médico'
            RETURN count(distinct c) as updated_convs
            """
            res = await session.run(tag_medical_convs)
            rec = await res.single()
            print(f"   -> Conversaciones marcadas como 'medical': {rec['updated_convs'] if rec else 0}")

            tag_physics_convs = """
            MATCH (c:Conversation)-[:CONTAINS]->(m:Message)
            WHERE (m.content =~ '(?i).*(física|fisica|newton|masa|aceleración|aceleracion|velocidad|cinemática|cinematica|dinámica|dinamica|resorte|fuerza|energía|energia|torque|socrático|socratico).*')
              AND (c.agent_id IS NULL OR NOT c.session_id STARTS WITH 'physics:')
            SET c.agent_id = 'physics',
                c.agent_name = 'Tutor Socrático de Física Multimodal'
            RETURN count(distinct c) as updated_convs
            """
            res = await session.run(tag_physics_convs)
            rec = await res.single()
            print(f"   -> Conversaciones marcadas como 'physics': {rec['updated_convs'] if rec else 0}")

            # 2. Tagging de entidades asociadas a conversaciones marcadas
            print("\n📌 Vinculando entidades a sus respectivos agentes...")
            tag_medical_entities = """
            MATCH (c:Conversation {agent_id: 'medical'})-[:CONTAINS]->(m:Message)-[:MENTIONS]->(e:Entity)
            SET e.agent_id = 'medical',
                e.agent_name = 'Asistente Médico'
            RETURN count(distinct e) as updated_entities
            """
            res = await session.run(tag_medical_entities)
            rec = await res.single()
            print(f"   -> Entidades vinculadas a 'medical' por mención: {rec['updated_entities'] if rec else 0}")

            tag_physics_entities = """
            MATCH (c:Conversation {agent_id: 'physics'})-[:CONTAINS]->(m:Message)-[:MENTIONS]->(e:Entity)
            SET e.agent_id = 'physics',
                e.agent_name = 'Tutor Socrático de Física Multimodal'
            RETURN count(distinct e) as updated_entities
            """
            res = await session.run(tag_physics_entities)
            rec = await res.single()
            print(f"   -> Entidades vinculadas a 'physics' por mención: {rec['updated_entities'] if rec else 0}")

            # 3. Tagging léxico para entidades médicas / histológicas que pudieran haber quedado huérfanas
            tag_orphan_medical = """
            MATCH (e:Entity)
            WHERE e.agent_id IS NULL
              AND (e.name =~ '(?i).*(arteria|célula|celula|tejido|endotelio|epitelio|histolog|túnica|tunica|adventicia|capilar|vena|vena|linfátic|linfatic|músculo|musculo|mucosa|estroma).*'
                   OR (e.description IS NOT NULL AND e.description =~ '(?i).*(histolog|arteria|célula|tejido|endotelio).*'))
            SET e.agent_id = 'medical',
                e.agent_name = 'Asistente Médico'
            RETURN count(e) as updated
            """
            res = await session.run(tag_orphan_medical)
            rec = await res.single()
            print(f"   -> Entidades médicas huérfanas clasificadas: {rec['updated'] if rec else 0}")

            # 4. Tagging léxico para entidades de física que pudieran haber quedado huérfanas
            tag_orphan_physics = """
            MATCH (e:Entity)
            WHERE e.agent_id IS NULL
              AND (e.name =~ '(?i).*(newton|fuerza|resorte|masa|cinemática|cinematica|velocidad|aceleración|aceleracion|energía|energia|torque|dinámica|dinamica|gravedad|fricción|friccion|trabajo).*'
                   OR (e.description IS NOT NULL AND e.description =~ '(?i).*(física|fisica|newton|cinemática|dinámica).*'))
            SET e.agent_id = 'physics',
                e.agent_name = 'Tutor Socrático de Física Multimodal'
            RETURN count(e) as updated
            """
            res = await session.run(tag_orphan_physics)
            rec = await res.single()
            print(f"   -> Entidades de física huérfanas clasificadas: {rec['updated'] if rec else 0}")

            # 5. Resumen de entidades por agent_id
            print("\n📊 Resumen actual de entidades en Neo4j por agent_id:")
            summary_query = """
            MATCH (e:Entity)
            RETURN coalesce(e.agent_id, 'UNASSIGNED') as agent, count(e) as total
            ORDER BY total DESC
            """
            res = await session.run(summary_query)
            records = await res.data()
            for r in records:
                print(f"   - Agente: {r['agent']:15} | Total entidades: {r['total']}")

    print("\n✅ Proceso de etiquetado y partición completado.")


if __name__ == "__main__":
    asyncio.run(tag_existing_nodes())
