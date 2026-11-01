# ==============================================================================
# Open Policy Agent (OPA) - Rego Policy: nams.authz
# Control de acceso determinista para mutaciones en el KG canónico (Dual-Rol)
# ==============================================================================

package nams.authz

import rego.v1

# Por defecto, ninguna mutación al KG canónico está permitida (Zero-Trust)
default allow_kg_mutation := false
default allow_misconception_candidate := false

# 1. Autorización de Mutación Canónica:
# Solo permitida si el modo es estrictamente 'professor' y no es una sesión de alumno.
allow_kg_mutation if {
    input.mode == "professor"
    input.user_id != ""
    not input.is_student_session
}

# 2. Denegación Explícita de Alumnos:
# Las afirmaciones de alumnos jamás alteran los conceptos canónicos.
deny_student_mutation if {
    input.mode == "student"
}

# 3. Autorización para Registrar Candidatos a Misconception:
# Permitida en modo 'student' cuando existe student_id y severidad válida.
allow_misconception_candidate if {
    input.mode == "student"
    input.student_id != ""
    input.concept != ""
    input.severity >= 1.0
}

