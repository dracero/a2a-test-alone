"""LangSmith Custom Evaluators for the Medical Images Agent.

This script defines custom evaluators for:
- Context Relevance (evaluating if retrieved medical text/images are relevant to the medical query)
- Context Recall (evaluating if retrieved medical text/images cover the reference diagnosis)

It also includes a harness to run these evaluators on a LangSmith dataset.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add workspace root to python path to import api_key_rotator
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(WORKSPACE_ROOT))

from api_key_rotator import create_google_llm, invoke_with_retry
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import evaluate, Client
from langsmith.evaluation import RunEvaluator, EvaluationResult

# Load environment variables
load_dotenv(dotenv_path=WORKSPACE_ROOT / '.env')


class MedicalContextRelevanceEvaluator(RunEvaluator):
    """Custom LangSmith evaluator to measure Context Relevance for the Medical Agent."""
    
    def __init__(self):
        # Initialize Google Gemini 2.5 Flash as the evaluator judge
        self.llm = create_google_llm(model="gemini-2.5-flash", temperature=0.0)

    def evaluate_run(self, run, example=None, **kwargs) -> EvaluationResult:
        query = run.inputs.get("consulta_usuario") or run.inputs.get("query")
        
        # 1. Traverse child runs to find the retriever (qdrant_medical_vector_search)
        retrieved_contexts = []
        if run.child_runs:
            for child in run.child_runs:
                if child.run_type == "retriever" or "search" in child.name.lower() or "buscar" in child.name.lower():
                    # Extraer documentos y figuras recuperados de Qdrant
                    if child.outputs and isinstance(child.outputs, tuple) and len(child.outputs) > 0:
                        points = child.outputs[0]  # buscar_muvera_2stage retorna (resultados, has_rejected)
                        if isinstance(points, list):
                            for doc in points:
                                payload = doc.get("payload", {})
                                text = payload.get("text", "")
                                caption = payload.get("caption", "")
                                path = payload.get("imagen_path", "")
                                if text:
                                    retrieved_contexts.append(f"[Texto]: {text}")
                                if caption:
                                    retrieved_contexts.append(f"[Figura]: {caption} (Ruta: {path})")
        
        # Fallback to general outputs
        if not retrieved_contexts:
            doc_ctx = run.outputs.get("contexto_documentos") or run.outputs.get("context", "")
            if isinstance(doc_ctx, str) and doc_ctx.strip():
                retrieved_contexts.append(doc_ctx)

        context_str = "\n---\n".join(retrieved_contexts).strip()
        if not context_str:
            return EvaluationResult(
                key="context_relevance", 
                score=0.0, 
                comment="No se recuperó ningún contexto médico de Qdrant."
            )

        # 2. Build LLM prompt to grade relevance
        system_prompt = (
            "Eres un evaluador experto en patología y diagnóstico médico. Tu tarea es calificar si el Contexto Médico Recuperado "
            "contiene información histopatológica o clínica relevante, descripción de figuras médicas y textos anatómicos "
            "necesarios para orientar el diagnóstico de la Consulta Médica del usuario.\n"
            "Califica con un score numérico decimal entre 0.0 (completamente irrelevante/ruido) "
            "y 1.0 (altamente relevante, contiene todo lo necesario para diagnosticar).\n"
            "Responde únicamente en formato JSON con la siguiente estructura:\n"
            "{\n"
            "  \"score\": <float_entre_0_y_1>,\n"
            "  \"comment\": \"<explicación_corta_del_motivo_del_score>\"\n"
            "}"
        )
        user_prompt = f"Consulta Médica:\n{query}\n\nContexto Médico Recuperado:\n{context_str}"
        
        try:
            res = invoke_with_retry(
                self.llm, 
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            content = res.content.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(content)
            return EvaluationResult(
                key="context_relevance",
                score=float(data.get("score", 0.0)),
                comment=data.get("comment", "")
            )
        except Exception as e:
            return EvaluationResult(key="context_relevance", score=0.0, comment=f"Error evaluando relevancia: {e}")


class MedicalContextRecallEvaluator(RunEvaluator):
    """Custom LangSmith evaluator to measure Context Recall for the Medical Agent."""
    
    def __init__(self):
        self.llm = create_google_llm(model="gemini-2.5-flash", temperature=0.0)

    def evaluate_run(self, run, example=None, **kwargs) -> EvaluationResult:
        # Get ground truth diagnosis response
        ground_truth = (
            example.outputs.get("respuesta_final") or 
            example.outputs.get("output") or 
            example.outputs.get("diagnosis")
        )
        
        if not ground_truth:
            return EvaluationResult(
                key="context_recall", 
                score=0.0, 
                comment="No hay un informe de referencia (Ground Truth) en el dataset."
            )

        # Traverse child runs to find the retriever
        retrieved_contexts = []
        if run.child_runs:
            for child in run.child_runs:
                if child.run_type == "retriever" or "search" in child.name.lower() or "buscar" in child.name.lower():
                    if child.outputs and isinstance(child.outputs, tuple) and len(child.outputs) > 0:
                        points = child.outputs[0]
                        if isinstance(points, list):
                            for doc in points:
                                payload = doc.get("payload", {})
                                text = payload.get("text", "")
                                caption = payload.get("caption", "")
                                if text:
                                    retrieved_contexts.append(f"[Texto]: {text}")
                                if caption:
                                    retrieved_contexts.append(f"[Figura]: {caption}")
        
        if not retrieved_contexts:
            doc_ctx = run.outputs.get("contexto_documentos") or run.outputs.get("context", "")
            if isinstance(doc_ctx, str) and doc_ctx.strip():
                retrieved_contexts.append(doc_ctx)

        context_str = "\n---\n".join(retrieved_contexts).strip()
        if not context_str:
            return EvaluationResult(
                key="context_recall", 
                score=0.0, 
                comment="No se recuperó ningún contexto médico de Qdrant."
            )

        # Build LLM prompt to grade recall
        system_prompt = (
            "Eres un evaluador experto en patología y diagnóstico médico. Tu tarea es analizar el Informe/Diagnóstico de Referencia (Ground Truth) "
            "y determinar qué proporción de las conclusiones, observaciones histológicas y hallazgos patológicos clave de esa referencia "
            "están contenidos dentro del Contexto Médico Recuperado.\n"
            "Califica con un score numérico decimal entre 0.0 (ningún concepto clave del informe está en el contexto) "
            "y 1.0 (todos los conceptos clave del informe de referencia están cubiertos por el contexto).\n"
            "Responde únicamente en formato JSON con la siguiente estructura:\n"
            "{\n"
            "  \"score\": <float_entre_0_y_1>,\n"
            "  \"comment\": \"<explicación_corta_del_motivo_del_score>\"\n"
            "}"
        )
        user_prompt = f"Informe de Referencia (Ground Truth):\n{ground_truth}\n\nContexto Médico Recuperado:\n{context_str}"
        
        try:
            res = invoke_with_retry(
                self.llm, 
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            content = res.content.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(content)
            return EvaluationResult(
                key="context_recall",
                score=float(data.get("score", 0.0)),
                comment=data.get("comment", "")
            )
        except Exception as e:
            return EvaluationResult(key="context_recall", score=0.0, comment=f"Error evaluando recall: {e}")


# Helper function to run the evaluation
def run_medical_agent_evaluation(dataset_name: str = "medical-assistant-test-dataset"):
    """Runs evaluation on a LangSmith dataset using the custom evaluators."""
    from app.agent import SistemaRAGColPaliPuro
    import asyncio
    from langsmith.utils import LangSmithNotFoundError
    
    client = Client()
    
    # Check if dataset exists, if not create and populate it
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"📦 Dataset '{dataset_name}' encontrado en LangSmith.")
    except LangSmithNotFoundError:
        print(f"✨ Dataset '{dataset_name}' no encontrado. Creándolo en LangSmith...")
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Dataset para evaluar el agente de imágenes médicas."
        )
        
        # Inline medical diagnostic examples
        examples = [
            {
                "consulta_usuario": "Analizar caso de sospecha de adenocarcinoma gástrico y biomarcadores HER2.",
                "respuesta_final": "El análisis histopatológico revela adenocarcinoma gástrico moderadamente diferenciado con patrón de crecimiento tubular. Inmunohistoquímica (IHC) positiva para HER2 (score 3+), lo cual sugiere elegibilidad para terapia con trastuzumab."
            },
            {
                "consulta_usuario": "Paciente con biopsia hepática para descartar cirrosis y evaluar grado de fibrosis.",
                "respuesta_final": "La biopsia hepática con tinción de Tricrómico de Masson muestra puentes portoportales y portocentrales de tejido conectivo fibroso con distorsión de la arquitectura lobulillar, compatible con cirrosis hepática avanzada (Grado F4 según escala METAVIR)."
            }
        ]
        
        print("📂 Cargando ejemplos de prueba médicos por defecto...")
        for item in examples:
            client.create_example(
                inputs={"consulta_usuario": item["consulta_usuario"]},
                outputs={"respuesta_final": item["respuesta_final"]},
                dataset_id=dataset.id
            )
        print(f"✅ {len(examples)} ejemplos subidos exitosamente.")

    # 1. Initialize Medical Agent (SistemaRAGColPaliPuro)
    agent = SistemaRAGColPaliPuro()
    agent.inicializar_componentes()
    
    # 2. Define the target function for LangSmith evaluate
    def target(inputs: dict) -> dict:
        query = inputs.get("consulta_usuario") or inputs.get("query")
        # Run medical agent workflow synchronously
        loop = asyncio.get_event_loop()
        
        # Simulating run state for agent
        initial_state = {
            "consulta_usuario": query,
            "messages": [],
            "trayectoria": [],
            "filtros_ontologia": [],
            "resultados_busqueda": [],
            "imagenes_relevantes": [],
            "contexto_documentos": "",
            "respuesta_final": "",
            "requiere_imagen": False,
            "consulta_optimizada": "",
            "user_id": "langsmith-eval-user",
            "tiempo_inicio": 0.0,
            "abortar_reset": False,
            "imagen_base64": None,
            "imagen_consulta": None
        }
        
        # Ensure compiled graph is setup
        if not agent.compiled_graph:
            return {"output": "Error: Sistema RAG no inicializado."}
            
        result_state = loop.run_until_complete(agent.compiled_graph.ainvoke(initial_state))
        return {
            "output": result_state.get("respuesta_final", ""),
            "document_context": result_state.get("contexto_documentos", "")
        }

    # 3. Execute evaluation
    print(f"🚀 Iniciando evaluación en LangSmith del agente Médico sobre dataset '{dataset_name}'...")
    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[
            MedicalContextRelevanceEvaluator(),
            MedicalContextRecallEvaluator()
        ],
        experiment_prefix="medical-custom-eval"
    )
    print("⏳ Sincronizando resultados pendientes con LangSmith...")
    client.flush()
    print("✅ Evaluación completada. Los resultados ya están disponibles en tu consola de LangSmith.")
    return results


if __name__ == "__main__":
    # Si se ejecuta directamente, inicia la evaluación sobre un dataset por defecto
    dataset = os.getenv("LANGSMITH_MEDICAL_DATASET", "medical-assistant-test-dataset")
    run_medical_agent_evaluation(dataset)
