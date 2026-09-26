"""
Script para resetear e indexar los PDFs del Tutor Socrático de Física en Qdrant.

Acciones:
1. Limpia/elimina las colecciones de física en Qdrant (documentos_pdf_texto_hf, documentos_pdf_imagenes).
2. Lee y procesa todos los PDFs de samples/python/agents/multimodal/PDF/.
3. Genera embeddings (HuggingFace MiniLM-L12 para texto, CLIP para imágenes) y los almacena en Qdrant.
4. Genera/actualiza temario.txt y crea el marcador .pdf_processing_done.
5. Ejecuta una consulta de verificación de lectura sobre los fragmentos indexados.

Uso:
    uv run --project samples/python/agents/multimodal python scripts/reset_and_index_physics_pdfs.py
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parents[1]
load_dotenv(root_dir / ".env", override=True)

# Agregar paths necesarios
agent_dir = root_dir / "samples/python/agents/multimodal"
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(agent_dir))

from qdrant_client import AsyncQdrantClient
from app.agent import PhysicsMultimodalAgent


async def main():
    print("=" * 80)
    print("🚀 RESET E INDEXACIÓN DE PDFs - TUTOR SOCRÁTICO DE FÍSICA")
    print("=" * 80)

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    print(f"📡 Conectando a Qdrant en: {qdrant_url}")
    client = AsyncQdrantClient(url=qdrant_url, timeout=60.0)

    # 1. Resetear colecciones de física en Qdrant
    collections_to_reset = [
        "documentos_pdf_texto_hf",
        "documentos_pdf_imagenes",
        "documentos_multimodal",
    ]

    print("\n1️⃣ Limpiando colecciones anteriores en Qdrant...")
    for col_name in collections_to_reset:
        try:
            await client.delete_collection(col_name)
            print(f"   🗑️ Colección '{col_name}' eliminada correctamente.")
        except Exception:
            print(f"   ℹ️ Colección '{col_name}' no existía previamente.")

    # 2. Inicializar el agente de Física
    print("\n2️⃣ Inicializando PhysicsMultimodalAgent y cargando modelos de embeddings...")
    agent = PhysicsMultimodalAgent(qdrant_url=qdrant_url)

    # 3. Buscar PDFs
    pdf_dir = agent_dir / "PDF"
    pdf_files = sorted([str(f) for f in pdf_dir.glob("*.pdf")])
    print(f"\n3️⃣ Se encontraron {len(pdf_files)} archivos PDF en {pdf_dir}:")
    for f in pdf_files:
        print(f"   • {Path(f).name} ({Path(f).stat().st_size // 1024} KB)")

    if not pdf_files:
        print("❌ ERROR: No se encontraron archivos PDF para procesar.")
        return

    # 4. Procesar y almacenar en Qdrant
    print("\n4️⃣ Procesando PDFs, generando embeddings y almacenando en Qdrant...")
    temario = await agent.procesar_y_almacenar_pdfs(pdf_files)

    # 5. Marcar como completado
    marker = agent_dir / ".pdf_processing_done"
    marker.touch()
    print(f"\n✅ Marcador de procesamiento creado: {marker.name}")

    # 6. Verificación en Qdrant
    print("\n5️⃣ Verificando almacenamiento en Qdrant...")
    col_info = await client.get_collection(agent.text_collection)
    print(f"   📦 Colección '{agent.text_collection}':")
    print(f"      - Puntos indexados: {col_info.points_count}")
    print(f"      - Estado: {col_info.status}")

    # 7. Test de búsqueda semántica
    print("\n6️⃣ Prueba de lectura y búsqueda semántica de prueba...")
    test_query = "Tercera Ley de Newton principio de interacción fuerzas pares"
    print(f"   🔍 Query: '{test_query}'")
    search_results = await agent.search_multimodal(query=test_query, top_k=3)
    text_results = search_results.get("text", [])

    print(f"   📄 Fragmentos recuperados: {len(text_results)}")
    for i, r in enumerate(text_results):
        payload = r.get("payload", {})
        pdf_name = Path(payload.get("pdf_name", "Desconocido")).name
        score = r.get("score")
        text_snippet = payload.get("text", "")[:180].replace("\n", " ")
        print(f"      [{i+1}] {pdf_name} (score: {score}): \"{text_snippet}...\"")

    assert len(text_results) > 0, "❌ ERROR: No se pudieron recuperar fragmentos de texto de los PDFs!"

    print("\n" + "=" * 80)
    print("🎉 RESET E INDEXACIÓN COMPLETADOS CON ÉXITO")
    print(f"   El Tutor Socrático de Física ahora puede leer todos los {len(pdf_files)} PDFs.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
