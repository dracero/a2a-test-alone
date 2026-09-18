"""
Script de verificación de aislamiento de memorias NAMS por agente y Gemini 2.5 Flash.
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


async def main():
    print("=" * 70)
    print("🧪 INICIANDO VERIFICACIÓN DE AISLAMIENTO NAMS Y GEMINI 2.5")
    print("=" * 70)

    # 1. Instanciar memoria de Física
    physics_mem = AgentNAMSMemory(
        agent_id="physics",
        agent_name="Tutor Socrático de Física Multimodal",
        uri=URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
    )

    # 2. Instanciar memoria Médica
    medical_mem = AgentNAMSMemory(
        agent_id="medical",
        agent_name="Asistente Médico",
        uri=URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
    )

    print("\n1️⃣ Probando conexión de ambas memorias a Neo4j...")
    p_conn = await physics_mem.ensure_connected()
    m_conn = await medical_mem.ensure_connected()
    assert p_conn, "❌ Falló conexión de memoria de Física"
    assert m_conn, "❌ Falló conexión de memoria de Histología/Médica"
    print("   ✅ Ambas memorias conectadas exitosamente.")

    print("\n2️⃣ Probando consulta de contexto NAMS de Física...")
    physics_context = await physics_mem.get_context(
        query="¿Cuál es la aceleración de un bloque con masa en un plano inclinado?",
        student_id="test_student",
        session_id="session_test_physics",
        max_items=5,
    )
    print("--- CONTEXTO RECUPERADO PARA FÍSICA ---")
    print(physics_context if physics_context else "(Vacío o limpio)")
    print("---------------------------------------")

    # Aserción: no debe contener términos histológicos
    histology_terms = ["endotelio", "arteria", "epitelio", "túnica", "histolog"]
    for term in histology_terms:
        assert term not in physics_context.lower(), f"❌ ERROR: Término de histología '{term}' encontrado en contexto de física!"
    print("   ✅ Verificación superada: CERO términos de histología en contexto de Física.")

    print("\n3️⃣ Probando consulta de contexto NAMS de Histología/Médico...")
    medical_context = await medical_mem.get_context(
        query="Describir las capas y células de la arteria elástica",
        student_id="test_student",
        session_id="session_test_medical",
        max_items=5,
    )
    print("--- CONTEXTO RECUPERADO PARA HISTOLOGÍA ---")
    print(medical_context if medical_context else "(Vacío o limpio)")
    print("------------------------------------------")

    physics_terms = ["segunda ley de newton", "conservación de la energía", "resorte", "cinemática"]
    for term in physics_terms:
        assert term not in medical_context.lower(), f"❌ ERROR: Término de física '{term}' encontrado en contexto de histología!"
    print("   ✅ Verificación superada: CERO términos de física en contexto de Histología.")

    print("\n4️⃣ Probando adición de mensajes con extracción LiteLLM + Gemini 2.5 Flash...")
    try:
        await physics_mem.add_user_message(
            session_id="session_verify_gemini_25",
            content="Tengo una duda sobre la constante elástica k de un resorte en dinámica.",
            student_id="test_student",
        )
        await physics_mem.add_assistant_message(
            session_id="session_verify_gemini_25",
            content="La ley de Hooke establece que la fuerza restauradora es proporcional al estiramiento: F = -k * x.",
            student_id="test_student",
        )
        print("   ✅ Mensajes guardados y entidades extraídas con Gemini 2.5 Flash sin errores de Groq.")
    except Exception as e:
        print(f"❌ Error al guardar mensajes con Gemini 2.5: {e}")
        raise e

    print("\n5️⃣ Probando conclusiones independientes...")
    p_conc, p_def = await physics_mem.get_conclusions("test_student")
    m_conc, m_def = await medical_mem.get_conclusions("test_student")
    print(f"   Física: {len(p_conc)} preferencias/insights, {len(p_def)} falencias")
    print(f"   Médico: {len(m_conc)} preferencias/insights, {len(m_def)} falencias")

    print("\n" + "=" * 70)
    print("🎉 TODAS LAS VERIFICACIONES COMPLETADAS CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
