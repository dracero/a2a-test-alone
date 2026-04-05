# Multimodal Physics Agent

Este agente actúa como un profesor de física experto (estilo UBA), capaz de responder preguntas analizando contexto (documentos/temario) e imágenes. Implementa un modo de **diálogo socrático** y renderiza ecuaciones matemáticas en formato LaTeX.

## DSPy GEPA Prompt Optimization

Este proyecto usa **dspy.GEPA (Genetic-Evolutionary Prompt Architecture)** para evolucionar las instrucciones del sistema y alinear matemáticamente el tono del agente con ejemplos reales del profesor de física mediante *reflective prompt evolution*.

Dado que este optimizador exige DSPy >= 3.0.0 y el workspace principal (`a2a-test-alone`) tiene dependencias de versiones anteriores arraigadas a través de `crewai`, la optimización **debe ejecutarse en su propio entorno virtual aislado**.

### 1. Activar el Entorno Aislado

```bash
cd samples/python/agents/multimodal

# Crear el entorno aislado de uv (ya creado)
uv venv

# Instalar dependencias puras (sin resolver el workspace)
uv pip install -e .

# Usar SIEMPRE el python de este entorno para evitar conflictos
```

### 2. Ejecutar la Optimización GEPA

```bash
# Prueba rápida (Dry Run) - Asegura que el LLM y DSPy funcionen
.venv/bin/python optimize_prompts.py --dry-run

# Optimización ligth (usa pocas métricas/tokens)
.venv/bin/python optimize_prompts.py --budget light

# Optimización completa (mayor exploración en el árbol de evolución de prompts)
.venv/bin/python optimize_prompts.py --budget medium
```
Los prompts optimizados se guardan tanto en `optimized_prompts.json` (estructuras JSON) como en un módulo compilado `.dspy`.

### 3. Evaluar los Prompts

Puede comparar las métricas precisas del _baseline_ frente al _modelo instruido por GEPA_:

```bash
.venv/bin/python evaluate_prompts.py
```
