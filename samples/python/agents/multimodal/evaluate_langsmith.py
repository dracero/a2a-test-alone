"""LangSmith Custom Evaluators for the Physics Multimodal Tutor Agent.

This script defines custom evaluators for:
- Context Relevance (evaluating if retrieved PDF context is relevant to the physics query)
- Context Recall (evaluating if retrieved PDF context covers the reference ground truth response)

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


class PhysicsContextRelevanceEvaluator(RunEvaluator):
    """Custom LangSmith evaluator to measure Context Relevance for the Physics Agent."""
    
    def __init__(self):
        # Initialize Google Gemini 2.5 Flash as the evaluator judge
        self.llm = create_google_llm(model="gemini-2.5-flash", temperature=0.0)

    def evaluate_run(self, run, example=None, **kwargs) -> EvaluationResult:
        query = run.inputs.get("query")
        
        # 1. Traverse child runs to find the retriever run (search_qdrant)
        retrieved_texts = []
        if run.child_runs:
            for child in run.child_runs:
                if child.run_type == "retriever" or "search" in child.name.lower():
                    if child.outputs and "text" in child.outputs:
                        for doc in child.outputs["text"]:
                            payload = doc.get("payload", {})
                            text = payload.get("text", "")
                            if text:
                                retrieved_texts.append(text)
        
        # Fallback to general run outputs if child runs aren't traced/logged
        if not retrieved_texts:
            doc_ctx = run.outputs.get("document_context") or run.outputs.get("context", "")
            if isinstance(doc_ctx, str) and doc_ctx.strip():
                retrieved_texts.append(doc_ctx)

        context_str = "\n---\n".join(retrieved_texts).strip()
        if not context_str:
            return EvaluationResult(
                key="context_relevance", 
                score=0.0, 
                comment="No se recuperó ningún contexto de los PDFs de física."
            )

        # 2. Build LLM prompt to grade relevance
        system_prompt = (
            "Eres un evaluador experto en Física I. Tu tarea es calificar si el Contexto Recuperado "
            "de los PDFs del curso es relevante, contiene los conceptos teóricos adecuados y las ecuaciones "
            "necesarias para responder de manera completa a la Consulta del Estudiante.\n"
            "Califica con un score numérico decimal entre 0.0 (completamente irrelevante/ruido) "
            "y 1.0 (altamente relevante, contiene todo lo necesario para responder).\n"
            "Responde únicamente en formato JSON con la siguiente estructura:\n"
            "{\n"
            "  \"score\": <float_entre_0_y_1>,\n"
            "  \"comment\": \"<explicación_corta_del_motivo_del_score>\"\n"
            "}"
        )
        user_prompt = f"Consulta del Estudiante:\n{query}\n\nContexto Recuperado:\n{context_str}"
        
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


class PhysicsContextRecallEvaluator(RunEvaluator):
    """Custom LangSmith evaluator to measure Context Recall for the Physics Agent."""
    
    def __init__(self):
        self.llm = create_google_llm(model="gemini-2.5-flash", temperature=0.0)

    def evaluate_run(self, run, example=None, **kwargs) -> EvaluationResult:
        # Get ground truth response from dataset
        ground_truth = (
            example.outputs.get("professor_response") or 
            example.outputs.get("output") or 
            example.outputs.get("response")
        )
        
        if not ground_truth:
            return EvaluationResult(
                key="context_recall", 
                score=0.0, 
                comment="No hay una respuesta de referencia (Ground Truth) en el dataset."
            )

        # Traverse child runs to find the retriever run (search_qdrant)
        retrieved_texts = []
        if run.child_runs:
            for child in run.child_runs:
                if child.run_type == "retriever" or "search" in child.name.lower():
                    if child.outputs and "text" in child.outputs:
                        for doc in child.outputs["text"]:
                            payload = doc.get("payload", {})
                            text = payload.get("text", "")
                            if text:
                                retrieved_texts.append(text)
        
        if not retrieved_texts:
            doc_ctx = run.outputs.get("document_context") or run.outputs.get("context", "")
            if isinstance(doc_ctx, str) and doc_ctx.strip():
                retrieved_texts.append(doc_ctx)

        context_str = "\n---\n".join(retrieved_texts).strip()
        if not context_str:
            return EvaluationResult(
                key="context_recall", 
                score=0.0, 
                comment="No se recuperó ningún contexto de los PDFs de física."
            )

        # Build LLM prompt to grade recall
        system_prompt = (
            "Eres un evaluador experto en Física I. Tu tarea es analizar la Respuesta de Referencia (Ground Truth) "
            "que representa la explicación ideal del profesor, y determinar qué proporción de los hechos, conceptos "
            "físicos y ecuaciones clave de esa respuesta de referencia están contenidos dentro del Contexto Recuperado.\n"
            "Califica con un score numérico decimal entre 0.0 (ningún concepto clave de la respuesta está en el contexto) "
            "y 1.0 (todos los conceptos clave de la respuesta están en el contexto).\n"
            "Responde únicamente en formato JSON con la siguiente estructura:\n"
            "{\n"
            "  \"score\": <float_entre_0_y_1>,\n"
            "  \"comment\": \"<explicación_corta_del_motivo_del_score>\"\n"
            "}"
        )
        user_prompt = f"Respuesta de Referencia (Ground Truth):\n{ground_truth}\n\nContexto Recuperado:\n{context_str}"
        
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
def run_agent_evaluation(dataset_name: str = "physics-tutor-test-dataset"):
    """Runs evaluation on a LangSmith dataset using the custom evaluators."""
    from app.agent import PhysicsMultimodalAgent
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
            description="Dataset para evaluar el agente multimodal de física."
        )
        
        script_dir = Path(__file__).resolve().parent
        examples_file = script_dir / "training_examples.json"
        if examples_file.exists():
            print(f"📂 Cargando ejemplos desde {examples_file.name}...")
            with open(examples_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for item in data:
                client.create_example(
                    inputs={"query": item["question"]},
                    outputs={"professor_response": item["professor_response"]},
                    dataset_id=dataset.id
                )
            print(f"✅ {len(data)} ejemplos subidos exitosamente.")
        else:
            print(f"⚠️ No se encontró {examples_file.name}. Dataset creado vacío.")

    # 1. Initialize Physics Agent
    agent = PhysicsMultimodalAgent()
    
    # 2. Define the target function for LangSmith evaluate
    def target(inputs: dict) -> dict:
        query = inputs.get("question") or inputs.get("query")
        # Run physics agent invoke synchronously
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(agent.invoke(
            query=query, 
            context_id="langsmith-eval-session",
            images=None
        ))
        return {"output": response}

    # 3. Execute evaluation
    print(f"🚀 Iniciando evaluación en LangSmith del agente Física sobre dataset '{dataset_name}'...")
    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[
            PhysicsContextRelevanceEvaluator(),
            PhysicsContextRecallEvaluator()
        ],
        experiment_prefix="physics-custom-eval"
    )
    print("✅ Evaluación completada. Los resultados ya están disponibles en tu consola de LangSmith.")
    return results


if __name__ == "__main__":
    # Si se ejecuta directamente, inicia la evaluación sobre un dataset por defecto
    dataset = os.getenv("LANGSMITH_PHYSICS_DATASET", "physics-tutor-test-dataset")
    run_agent_evaluation(dataset)
