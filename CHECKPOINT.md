# 📌 Punto de Control: JEV + OPA + SHACL (Opción A Neo4j)

**Fecha:** 31 de Octubre de 2026  
**Rama:** `completo`  
**Último Commit de Código:** `38dcaec` (*harness*)

---

## 1. Estado Actual Consolidado ✅

Todo el trabajo previo sobre **JEV** y **Harness²** quedó completado, verificado y comiteado:
- **JEV System One (Reemplazo Integral de GliNER)**:
  - Clase `JEVEntityExtractor` en [`jev_service.py`](file:///run/media/dracero/DiscoMecanico1/AIProjects/a2a-test-alone/jev_service.py) y [`demo/ui/service/server/jev_service.py`](file:///run/media/dracero/DiscoMecanico1/AIProjects/a2a-test-alone/demo/ui/service/server/jev_service.py).
  - Patrón *Select & Verify instead of Generate* con 0 VRAM adicional en GPU.
  - Cero hardcoding: extracción dinámica de n-gramas cruzada con la ontología formal POLE+O.
  - Inyección en [`demo/ui/service/server/agent_nams.py`](file:///run/media/dracero/DiscoMecanico1/AIProjects/a2a-test-alone/demo/ui/service/server/agent_nams.py).
- **Google Research Harness² (`harness2`) en Formato Offline**:
  - Benchmark golden en [`data/harness_benchmarks/student_queries_benchmark.json`](file:///run/media/dracero/DiscoMecanico1/AIProjects/a2a-test-alone/data/harness_benchmarks/student_queries_benchmark.json).
  - Motor recursivo (*Propose $\to$ Probe $\to$ Compose*) en [`scripts/harness_offline_eval.py`](file:///run/media/dracero/DiscoMecanico1/AIProjects/a2a-test-alone/scripts/harness_offline_eval.py).
  - Configuración calibrada en [`config/nams_harness_config.json`](file:///run/media/dracero/DiscoMecanico1/AIProjects/a2a-test-alone/config/nams_harness_config.json).
  - Scripts ejecutables listos: [`scripts/eval_harness_baseline.sh`](file:///run/media/dracero/DiscoMecanico1/AIProjects/a2a-test-alone/scripts/eval_harness_baseline.sh) y [`scripts/optimize_and_apply_harness.sh`](file:///run/media/dracero/DiscoMecanico1/AIProjects/a2a-test-alone/scripts/optimize_and_apply_harness.sh).
  - Documentación detallada en [`README.md`](file:///run/media/dracero/DiscoMecanico1/AIProjects/a2a-test-alone/README.md).

---

## 2. Tarea en Curso (Para Reanudar) 🚧

**Requerimiento del Usuario:**
> *"Implementá JEV, OPA y SHACL con la opción A para Neo4J"*

### Diagnóstico Técnico Realizado antes de pausar:
1. **Neo4j Local**: Contenedor `neo4j-local` activo en puerto `7687` (versión `neo4j:5.20.0-community`).
2. **Neosemantics (`n10s`)**: La Opción A requiere el plugin de validación SHACL nativo en Cypher (`CALL n10s.validation.shacl.validate()`). Se comprobó que aún no está cargado en el contenedor.
3. **Binario Neosemantics**: Disponible en GitHub para Neo4j 5.20:
   `https://github.com/neo4j-labs/neosemantics/releases/download/5.20.0/neosemantics-5.20.0.jar`

---

## 3. Plan de Acción Inmediato al Reanudar 📋

1. **Instalación de n10s en Neo4j Local**:
   - Descargar `neosemantics-5.20.0.jar`.
   - Copiar al contenedor: `docker cp neosemantics-5.20.0.jar neo4j-local:/var/lib/neo4j/plugins/`.
   - Reiniciar contenedor y habilitar `dbms.security.procedures.unrestricted=n10s.*,apoc.*`.
2. **Definición de Shapes SHACL (POLE+O)**:
   - Crear [`shapes/nams_poleo_shapes.ttl`](file:///run/media/dracero/DiscoMecanico1/AIProjects/a2a-test-alone/shapes/nams_poleo_shapes.ttl) modelando:
     - `ConceptShape`: nombre, dominio, aristas `[:REQUIRES]` válidas.
     - `StudentDeficiencyShape`: severidad en rango $[1.0, 5.0]$, ocurrencias $\ge 1$, arista obligatoria `[:ABOUT_CONCEPT]`.
     - `ReasoningTraceShape`: traza inmutable y aristas `[:TOUCHED]`.
3. **Módulo de Gobernanza OPA (Rego)**:
   - Crear políticas en `policies/` para:
     - Autorización de escritura Dual-Rol (`nams.authz`: solo modo `professor` muta el KG).
     - Gobernanza de consolidación de falencias (`nams.pedagogy`: evalúa umbrales del Harness sin prompts).
   - Implementar evaluador de políticas en Python para invocar Rego de forma determinista y ultrarrápida.
4. **Integración en `AgentNAMSMemory`**:
   - Flujo: Query $\to$ JEV System One $\to$ OPA Policy Check $\to$ SHACL Graph Validation $\to$ Neo4j Commit.
