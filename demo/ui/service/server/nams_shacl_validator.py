"""
NAMS SHACL Validator (Option A: Neo4j Native via Neosemantics / n10s).
Provides W3C SHACL validation directly inside Neo4j for POLE+O entities:
- Concept
- StudentDeficiency
- MisconceptionCandidate
- ReasoningTrace
"""

import os
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_SHAPES_PATH = str(Path(__file__).resolve().parents[4] / "shapes" / "nams_poleo_shapes.ttl")
if not os.path.exists(DEFAULT_SHAPES_PATH):
    alt_cand = str(Path(__file__).resolve().parents[2] / "shapes" / "nams_poleo_shapes.ttl")
    if os.path.exists(alt_cand):
        DEFAULT_SHAPES_PATH = alt_cand
    else:
        root_cand = str(Path(os.getcwd()) / "shapes" / "nams_poleo_shapes.ttl")
        if os.path.exists(root_cand):
            DEFAULT_SHAPES_PATH = root_cand


async def _execute_write_generic(target: Any, query: str, params: Optional[dict[str, Any]] = None) -> Any:
    """Ejecuta una consulta de escritura soportando neo4j_agent_memory client, AsyncDriver o AsyncSession."""
    if hasattr(target, "execute_write") and callable(target.execute_write):
        return await target.execute_write(query, params or {})
    elif hasattr(target, "session") and callable(target.session):
        async with target.session() as s:
            res = await s.run(query, params or {})
            return await res.data()
    elif hasattr(target, "run") and callable(target.run):
        res = await target.run(query, params or {})
        return await res.data()
    raise AttributeError(f"Target object {type(target)} does not support async cypher execution")


async def _execute_read_generic(target: Any, query: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """Ejecuta una consulta de lectura soportando neo4j_agent_memory client, AsyncDriver o AsyncSession."""
    if hasattr(target, "execute_read") and callable(target.execute_read):
        return await target.execute_read(query, params or {})
    elif hasattr(target, "session") and callable(target.session):
        async with target.session() as s:
            res = await s.run(query, params or {})
            return await res.data()
    elif hasattr(target, "run") and callable(target.run):
        res = await target.run(query, params or {})
        return await res.data()
    raise AttributeError(f"Target object {type(target)} does not support async cypher execution")


async def ensure_shacl_configured_async(neo4j_client: Any, shapes_path: Optional[str] = None) -> bool:
    """Configura e importa las shapes SHACL en Neo4j usando n10s de forma idempotente (asíncrono)."""
    s_path = shapes_path or DEFAULT_SHAPES_PATH
    if not os.path.exists(s_path):
        logger.warning(f"⚠️ Archivo SHACL no encontrado en {s_path}")
        return False

    try:
        with open(s_path, "r", encoding="utf-8") as f:
            shapes_turtle = f.read()

        # 1. Asegurar constraint obligatorio de n10s
        await _execute_write_generic(
            neo4j_client,
            "CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS FOR (r:Resource) REQUIRE r.uri IS UNIQUE",
        )

        # 2. Inicializar graphconfig si no existe
        try:
            await _execute_write_generic(
                neo4j_client,
                "CALL n10s.graphconfig.init({handleVocabUris: 'IGNORE', handleMultivals: 'OVERWRITE'})",
            )
        except Exception:
            # Ya inicializado previamente
            pass

        # 3. Limpiar e importar las shapes de POLE+O
        await _execute_write_generic(neo4j_client, "CALL n10s.validation.shacl.dropShapes()")
        await _execute_write_generic(
            neo4j_client,
            "CALL n10s.validation.shacl.import.inline($shapes, 'Turtle')",
            {"shapes": shapes_turtle},
        )
        logger.info(f"📐 [SHACL - n10s] Shapes ontológicas POLE+O cargadas exitosamente desde {s_path}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Error al configurar SHACL en Neo4j: {e}")
        return False


def ensure_shacl_configured_sync(neo4j_driver: Any, shapes_path: Optional[str] = None) -> bool:
    """Configura e importa las shapes SHACL en Neo4j usando n10s de forma síncrona."""
    s_path = shapes_path or DEFAULT_SHAPES_PATH
    if not os.path.exists(s_path):
        logger.warning(f"⚠️ Archivo SHACL no encontrado en {s_path}")
        return False

    try:
        with open(s_path, "r", encoding="utf-8") as f:
            shapes_turtle = f.read()

        with neo4j_driver.session() as session:
            session.run(
                "CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS FOR (r:Resource) REQUIRE r.uri IS UNIQUE"
            )
            try:
                session.run(
                    "CALL n10s.graphconfig.init({handleVocabUris: 'IGNORE', handleMultivals: 'OVERWRITE'})"
                )
            except Exception:
                pass

            session.run("CALL n10s.validation.shacl.dropShapes()")
            session.run(
                "CALL n10s.validation.shacl.import.inline($shapes, 'Turtle')",
                {"shapes": shapes_turtle},
            )
            logger.info(f"📐 [SHACL - n10s Sync] Shapes POLE+O cargadas desde {s_path}")
            return True
    except Exception as e:
        logger.warning(f"⚠️ Error al configurar SHACL en Neo4j (sync): {e}")
        return False


async def validate_nams_graph_async(neo4j_client: Any) -> list[dict[str, Any]]:
    """Ejecuta la validación SHACL nativa en Neo4j y devuelve la lista de violaciones."""
    cypher = """
    CALL n10s.validation.shacl.validate()
    YIELD focusNode, nodeType, shapeId, propertyShape, offendingValue, resultPath, severity, resultMessage, customMessage
    RETURN focusNode, nodeType, shapeId, propertyShape, offendingValue, resultPath, severity, resultMessage, customMessage
    """
    try:
        rows = await _execute_read_generic(neo4j_client, cypher)
        violations = [dict(r) for r in rows] if rows else []
        if violations:
            logger.warning(f"🚨 [SHACL Violation] Detectadas {len(violations)} violaciones ontológicas en Neo4j:")
            for v in violations[:5]:
                logger.warning(
                    f"   - Nodo {v.get('nodeType')}: propiedad '{v.get('resultPath')}' "
                    f"-> {v.get('customMessage') or v.get('resultMessage')}"
                )
        else:
            logger.info("✅ [SHACL Validation] El grafo NAMS cumple al 100% con los contratos ontológicos W3C.")
        return violations
    except Exception as e:
        logger.error(f"❌ Error ejecutando validación SHACL en Neo4j: {e}")
        return []


def validate_nams_graph_sync(neo4j_driver: Any) -> list[dict[str, Any]]:
    """Ejecuta la validación SHACL nativa de forma síncrona."""
    cypher = """
    CALL n10s.validation.shacl.validate()
    YIELD focusNode, nodeType, shapeId, propertyShape, offendingValue, resultPath, severity, resultMessage, customMessage
    RETURN focusNode, nodeType, shapeId, propertyShape, offendingValue, resultPath, severity, resultMessage, customMessage
    """
    try:
        with neo4j_driver.session() as session:
            res = session.run(cypher)
            violations = [dict(r) for r in res]
            if violations:
                logger.warning(f"🚨 [SHACL Sync] {len(violations)} violaciones detectadas.")
            else:
                logger.info("✅ [SHACL Sync] Grafo NAMS 100% conforme.")
            return violations
    except Exception as e:
        logger.error(f"❌ Error en validación SHACL sync: {e}")
        return []
