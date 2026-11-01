# ==============================================================================
# Open Policy Agent (OPA) - Rego Policy: nams.pedagogy
# Gobernanza pedagógica determinista para consolidación de falencias (Harness²)
# ==============================================================================

package nams.pedagogy

import rego.v1

default promote_to_deficiency := false

# Regla de consolidación por acumulación semántica calibrada:
# Promueve un cluster de candidatos a StudentDeficiency si y solo si:
# 1. La cantidad de ocurrencias alcanza el umbral calibrado (ej. >= 5).
# 2. La similitud coseno media alcanza el umbral de proximidad (ej. >= 0.82).
# 3. La severidad calculada amerita persistencia pedagógica permanente (ej. >= 2.0).
promote_to_deficiency if {
    input.occurrences >= input.harness_config.deficiency_count_threshold
    input.similarity >= input.harness_config.deficiency_similarity_threshold
    input.severity >= 2.0
}

# Consolidación acelerada para falencias críticas de severidad extrema (ej. >= 4.5)
# con menor recurrencia requerida si la similitud semántica es altísima.
promote_to_deficiency if {
    input.severity >= 4.5
    input.occurrences >= 3
    input.similarity >= 0.88
}
