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

**¿Cómo funciona?**
GEPA (Genetic-Evolutionary Prompt Architecture) optimiza los prompts evaluando iterativamente:
1. Resuelve los ejemplos con las instrucciones actuales.
2. Un LLM "juez" reflexiona y critica las respuestas obtenidas.
3. El LLM propone mejoras textuales sobre las instrucciones basándose en las críticas.
4. Genera un árbol de variaciones de prompts, usando un enfoque de optimización Pareto para quedarse con aquellos prompts que rinden mejor de manera equilibrada en todos los ejemplos.

Para correr el optimizador de prompts, puedes usar diferentes "presupuestos" (budgets) de búsqueda:

```bash
# Prueba rápida (Dry Run) - Asegura que el LLM y DSPy funcionen (no optimiza)
.venv/bin/python optimize_prompts.py --dry-run

# Optimización ligera (rápida, hace pocos intentos/evaluaciones)
.venv/bin/python optimize_prompts.py --budget light

# Optimización media (recomendada, mayor exploración en el árbol de evolución de prompts)
.venv/bin/python optimize_prompts.py --budget medium
```
Los prompts optimizados se guardan tanto en `optimized_prompts.json` (estructuras JSON legibles) como en un módulo compilado `.dspy`.

### 3. Ajustando la Optimización (Tuning)

Si la optimización generada de forma automática no es la **deseada** (por ejemplo, el tono no es exactamente como el profesor, o falla en cómo usar LaTeX), puedes guiar y arreglar el optimizador ajustando tres factores principales:

1. **Editar los Ejemplos de Entrenamiento (`training_examples.json`)**
   Esta es la parte más importante. Modifica los valores de `professor_response` en el JSON para que reflejen **exactamente** la forma en la que quieres que el modelo conteste. GEPA evolucionará las instrucciones intentando emular este texto (tono, estructura, humor, rigor). Usa pocos ejemplos (5-10) pero de calidad exquisita.

2. **Modificar la Métrica (Juez) en `optimize_prompts.py`**
   Ve a la función `professor_style_metric()` en el código. 
   - Puedes cambiar los **pesos** (`DIMENSION_WEIGHTS`) para darle más importancia matemática al `tono` o al `rigor`.
   - Puedes editar las instrucciones al LM evaluador en el texto del `judge_prompt`. Si quieres que castigue más los errores de LaTeX, indícaselo explícitamente en el texto. El juez devuelve feedback en formato de texto; este texto será lo que la etapa de "reflexión" lea para corregir los prompts iniciales.

3. **Ajustar Parámetros de Exploración por Terminal**
   Puedes modificar parámetros avanzados al correr el script:
   ```bash
   # --reflection-temperature más alta (ej: 1.2) provoca modificaciones más drásticas y creativas sobre las instrucciones
   .venv/bin/python optimize_prompts.py --budget medium --reflection-temperature 1.2

   # Usar un modelo potente exclusivo para reflexionar/evaluar mejora drásticamente las críticas
   .venv/bin/python optimize_prompts.py --budget medium --model mi-modelo-rapido --reflection-model modelo-mas-grande-y-detallado
   ```

### 4. Evaluar los Prompts

Puede comparar las métricas precisas del _baseline_ frente al _modelo instruido por GEPA_:

```bash
.venv/bin/python evaluate_prompts.py
```
