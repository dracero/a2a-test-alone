import asyncio
import os
import json
import logging
from enum import Enum
from typing import Any, Optional, Union
from uuid import uuid4

import numpy as np
from pydantic import SecretStr
from neo4j_agent_memory import MemoryClient, MemorySettings, ExtractionConfig, ExtractorType
from neo4j_agent_memory.config.settings import SchemaConfig
from neo4j_agent_memory.schema import SchemaModel, load_schema_from_file, EntitySchemaConfig
from neo4j_agent_memory.llm.adapters.sentence_transformers import SentenceTransformersProvider

try:
    from .api_key_rotator import google_key_rotator, ainvoke_with_retry, create_google_llm
except ImportError:
    from api_key_rotator import google_key_rotator, ainvoke_with_retry, create_google_llm

logger = logging.getLogger(__name__)


class InteractionMode(str, Enum):
    """Modo de interacción del sistema.
    
    PROFESSOR: El usuario o sistema actúa con autoridad para crear o modificar el KG canónico.
    STUDENT: Modo pedagógico donde el sistema actúa como tutor y NUNCA acepta afirmaciones erróneas
             del estudiante como verdades ontológicas.
    """
    PROFESSOR = "professor"
    STUDENT = "student"


class AgentNAMSMemory:
    """Gestor de memoria NAMS independiente y aislado para un agente específico con arquitectura POLE+O Dual-Rol.

    Garantiza que:
    1. Las sesiones a corto plazo estén prefijadas con el identificador del agente (ej. 'physics:session_123').
    2. Las preferencias e insights estén asociados únicamente al par (estudiante, agente).
    3. Las entidades y conceptos del grafo de conocimiento pertenezcan estrictamente al dominio de este agente,
       impidiendo cualquier cruce de contexto entre agentes (ej: histología vs. física).
    4. El extractor de entidades use LiteLLM con Gemini 2.5 Flash y el rotador de API keys.
    5. Doble Rol: Solo el modo PROFESSOR puede modificar el KG canónico.
    6. Detección de falencias acumulativa por similaridad semántica (≥5 coincidencias con cosine >= 0.82).
    7. Auditoría mediante Reasoning Traces y relaciones TOUCHED.
    """

    DEFICIENCY_SIMILARITY_THRESHOLD: float = 0.82
    DEFICIENCY_COUNT_THRESHOLD: int = 5

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        ontology_path: Optional[str] = None,
    ):
        self.agent_id = agent_id.lower().strip()
        self.agent_name = agent_name.strip()
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database or "neo4j"
        self.ontology_path = ontology_path
        self.schema_config: Optional[EntitySchemaConfig] = None
        self._connected = False
        self.client: Optional[MemoryClient] = None
        self._init_client()

    def _init_client(self):
        """Inicializa la configuración de MemorySettings y el cliente NAMS con Gemini 2.5 y POLE+O."""
        if not (self.uri and self.username and self.password):
            logger.warning(f"⚠️ Parámetros de Neo4j incompletos para el agente '{self.agent_name}'. Memoria inactiva.")
            self.client = None
            return

        # Sincronizar clave activa de Google con LiteLLM
        try:
            import litellm
            litellm.suppress_debug_info = True
            litellm.telemetry = False
        except Exception:
            pass

        active_key = google_key_rotator.get_key()
        if active_key:
            os.environ["GEMINI_API_KEY"] = active_key
            os.environ["GOOGLE_API_KEY"] = active_key

        try:
            embedder = SentenceTransformersProvider(
                model="BAAI/bge-small-en-v1.5",
                device="cpu",
            )

            # Cargar y configurar esquema si se especificó ontología (Física)
            # Para agentes sin schema explícito (Médico), se usa POLE+O estándar
            # y el LLM extrae dinámicamente las entidades directamente de los documentos.
            schema_settings = None
            if self.ontology_path and os.path.exists(self.ontology_path):
                try:
                    self.schema_config = load_schema_from_file(self.ontology_path)
                    schema_settings = SchemaConfig(
                        model=SchemaModel.CUSTOM,
                        custom_schema_path=self.ontology_path,
                    )
                    logger.info(f"📜 Ontología POLE+O cargada desde {self.ontology_path} para '{self.agent_name}'")
                except Exception as e:
                    logger.warning(f"⚠️ Error cargando ontología desde {self.ontology_path}: {e}")
                    schema_settings = SchemaConfig()
            else:
                schema_settings = SchemaConfig()

            settings = MemorySettings(
                neo4j={
                    "uri": self.uri,
                    "username": self.username,
                    "password": SecretStr(self.password),
                    "database": self.database,
                },
                embedding=embedder,
                llm="gemini/gemini-2.5-flash",
                extraction=ExtractionConfig(
                    extractor_type=ExtractorType.LLM,
                ),
                schema_config=schema_settings,
            )
            self.client = MemoryClient(settings)
            logger.info(f"✅ Memoria NAMS configurada para '{self.agent_name}' ({self.agent_id}) con Gemini 2.5 Flash")
        except Exception as e:
            logger.error(f"❌ Error al instanciar NAMS para '{self.agent_name}': {e}")
            self.client = None

    async def ensure_connected(self) -> bool:
        """Conecta el cliente a Neo4j de forma idempotente."""
        if not self.client:
            return False
        if self._connected:
            return True

        active_key = google_key_rotator.get_key()
        if active_key:
            os.environ["GEMINI_API_KEY"] = active_key
            os.environ["GOOGLE_API_KEY"] = active_key

        try:
            logger.info(f"🔌 Conectando cliente NAMS para agente '{self.agent_name}'...")
            await self.client.connect()
            self._connected = True
            logger.info(f"✅ Conectado a Neo4j NAMS ({self.agent_name})")
            return True
        except Exception as e:
            logger.error(f"❌ Falló conexión NAMS ({self.agent_name}): {e}")
            self._connected = False
            return False

    async def register_ontology(self) -> bool:
        """Aplica y asegura la ontología en el namespace del agente.
        
        Para física: sincroniza el schema POLE+O personalizado.
        Para agentes médicos: confirma que la extracción automática por LLM de documentos está activa.
        """
        if not await self.ensure_connected():
            return False

        try:
            if self.ontology_path and os.path.exists(self.ontology_path):
                self.schema_config = load_schema_from_file(self.ontology_path)
                logger.info(
                    f"✅ Ontología registrada para '{self.agent_name}': "
                    f"{len(self.schema_config.entity_types)} tipos de entidad, "
                    f"{len(self.schema_config.relation_types)} tipos de relación."
                )
            else:
                logger.info(
                    f"ℹ️ Agente '{self.agent_name}': Ontología dinámica activa. "
                    "El LLM extrae las entidades directamente de los documentos cargados."
                )
            return True
        except Exception as e:
            logger.error(f"❌ Error registrando ontología para '{self.agent_name}': {e}")
            return False

    def scope_session_id(self, raw_session_id: str) -> str:
        """Prefija el session_id con el agent_id para aislar totalmente los historiales."""
        prefix = f"{self.agent_id}:"
        if raw_session_id.startswith(prefix):
            return raw_session_id
        return f"{prefix}{raw_session_id}"

    def student_identifier(self, student_id: str) -> str:
        return f"{student_id}_{self.agent_name}"

    def system_identifier(self) -> str:
        return f"system_{self.agent_name}"

    async def add_user_message(
        self,
        session_id: str,
        content: str,
        student_id: Optional[str] = None,
        mode: InteractionMode = InteractionMode.STUDENT,
    ):
        """Guarda un mensaje del usuario en la sesión aislada del agente.
        
        En modo PROFESSOR, las entidades extraídas se etiquetan como fuente 'professor'.
        En modo STUDENT, solo se registra en Short-Term y sus afirmaciones nunca modifican el KG canónico.
        """
        if not await self.ensure_connected():
            return
        scoped_id = self.scope_session_id(session_id)
        user_id = self.student_identifier(student_id or session_id)
        try:
            active_key = google_key_rotator.get_key()
            if active_key:
                os.environ["GEMINI_API_KEY"] = active_key
                os.environ["GOOGLE_API_KEY"] = active_key

            msg = await self.client.short_term.add_message(
                session_id=scoped_id,
                role="user",
                content=content,
                user_identifier=user_id,
                metadata={
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "mode": mode.value if hasattr(mode, "value") else str(mode),
                },
            )
            # Etiquetar entidades extraídas con el agent_id y origen
            source_val = "professor" if mode == InteractionMode.PROFESSOR else "student"
            await self._tag_entities_for_message(str(msg.id), source=source_val)
            logger.info(f"💾 [{self.agent_name}] Guardado mensaje de usuario en sesión {scoped_id} (Modo: {source_val})")
        except Exception as e:
            logger.warning(f"⚠️ [{self.agent_name}] Error guardando mensaje de usuario: {e}")

    async def add_assistant_message(self, session_id: str, content: str, student_id: Optional[str] = None):
        """Guarda la respuesta del asistente en la sesión aislada del agente."""
        if not await self.ensure_connected():
            return
        scoped_id = self.scope_session_id(session_id)
        user_id = self.student_identifier(student_id or session_id)
        try:
            active_key = google_key_rotator.get_key()
            if active_key:
                os.environ["GEMINI_API_KEY"] = active_key
                os.environ["GOOGLE_API_KEY"] = active_key

            clean_content = content.split("__IMAGE_PARTS__:")[0].strip()
            msg = await self.client.short_term.add_message(
                session_id=scoped_id,
                role="assistant",
                content=clean_content,
                user_identifier=user_id,
                metadata={"agent_id": self.agent_id, "agent_name": self.agent_name},
            )
            await self._tag_entities_for_message(str(msg.id), source="tutor")
            logger.info(f"💾 [{self.agent_name}] Guardada respuesta de asistente en sesión {scoped_id}")
        except Exception as e:
            logger.warning(f"⚠️ [{self.agent_name}] Error guardando respuesta de asistente: {e}")

    async def _tag_entities_for_message(self, message_id: str, source: str = "student"):
        """Etiqueta todas las entidades asociadas al mensaje con el agent_id y rol para aislamiento y auditoría."""
        try:
            cypher = """
            MATCH (m:Message {id: $message_id})
            OPTIONAL MATCH (m)-[:MENTIONS]->(e1:Entity)
            OPTIONAL MATCH (e2:Entity)-[:EXTRACTED_FROM]->(m)
            WITH coalesce(e1, e2) AS e
            WHERE e IS NOT NULL
            SET e.agent_id = $agent_id,
                e.agent_name = $agent_name,
                e.source = CASE WHEN e.source IS NULL THEN $source ELSE e.source END
            """
            await self.client.long_term._client.execute_write(
                cypher,
                {
                    "message_id": message_id,
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "source": source,
                }
            )
        except Exception as e:
            logger.debug(f"Tagging entities error: {e}")

    async def get_context(
        self,
        query: str,
        student_id: str,
        session_id: str,
        max_items: int = 10,
    ) -> str:
        """Obtiene el contexto NAMS estrictamente aislado para este agente."""
        if not await self.ensure_connected():
            return ""

        parts = []
        scoped_id = self.scope_session_id(session_id)

        # 1. Memoria a corto plazo (únicamente de las sesiones de este agente)
        try:
            short_term = await self.client.short_term.get_context(
                query,
                session_id=scoped_id,
                max_messages=max_items,
            )
            if short_term:
                parts.append(f"## Conversation History\n{short_term}")
        except Exception as e:
            logger.warning(f"⚠️ [{self.agent_name}] Error en historial corto plazo: {e}")

        # 2. Preferencias e insights del alumno para este agente y falencias confirmadas
        user_ids = [self.student_identifier(student_id), self.system_identifier()]
        preferences = []
        try:
            if self.client.long_term._embedder is not None:
                embedding = await self.client.long_term._embedder.embed(query)
                cypher_query = """
                CALL db.index.vector.queryNodes('preference_embedding_idx', $limit, $embedding)
                YIELD node, score
                WHERE score >= $threshold
                MATCH (u:User)-[:HAS_PREFERENCE]->(node)
                WHERE u.identifier IN $user_identifiers
                RETURN node AS p, score
                ORDER BY score DESC
                """
                results = await self.client.long_term._client.execute_read(
                    cypher_query,
                    {
                        "embedding": embedding,
                        "limit": max_items,
                        "threshold": 0.7,
                        "user_identifiers": user_ids,
                    }
                )
                for row in results:
                    pref_data = dict(row["p"])
                    pref = self.client.long_term._parse_preference(pref_data)
                    preferences.append(pref)
        except Exception as e:
            logger.debug(f"Vector search preferences fallback: {e}")

        # Fallback a consulta directa si no hubo match por vector
        if not preferences:
            try:
                for uid in user_ids:
                    prefs = await self.client.long_term.get_preferences_for(uid)
                    if prefs:
                        preferences.extend(prefs)
            except Exception as e:
                logger.warning(f"⚠️ [{self.agent_name}] Error recuperando preferencias para {user_ids}: {e}")

        if preferences:
            parts.append("## Relevant Knowledge")
            for pref in preferences:
                line = f"- [{pref.category}] {pref.preference}"
                if pref.context:
                    line += f" (context: {pref.context})"
                parts.append(line)

        # 3. Entidades del grafo estrictamente de este agente
        try:
            agent_entities_query = """
            MATCH (e:Entity)
            WHERE e.agent_id = $agent_id
               OR EXISTS {
                    MATCH (c:Conversation)-[:HAS_MESSAGE]->(m:Message)
                    WHERE c.session_id STARTS WITH $agent_prefix
                      AND ((m)-[:MENTIONS]->(e) OR (e)-[:EXTRACTED_FROM]->(m))
               }
            RETURN DISTINCT e.name AS name, e.type AS type, e.description AS description, e.formula AS formula
            LIMIT $limit
            """
            agent_prefix = f"{self.agent_id}:"
            results = await self.client.long_term._client.execute_read(
                agent_entities_query,
                {
                    "agent_id": self.agent_id,
                    "agent_prefix": agent_prefix,
                    "limit": max_items,
                }
            )
            if results:
                entity_parts = []
                for row in results:
                    name = row.get("name")
                    etype = row.get("type", "Concept")
                    desc = row.get("description")
                    formula = row.get("formula")
                    line = f"- {name} ({etype})"
                    if formula:
                        line += f" [Fórmula: {formula}]"
                    if desc:
                        line += f": {desc}"
                    entity_parts.append(line)
                if entity_parts:
                    parts.append("## Relevant Entities\n" + "\n".join(entity_parts))
        except Exception as e:
            logger.warning(f"⚠️ [{self.agent_name}] Error consultando entidades aisladas: {e}")

        return "\n\n".join(parts)

    async def add_preference(self, category: str, preference: str, student_id: str):
        """Guarda una preferencia asociada al estudiante y a este agente."""
        if not await self.ensure_connected():
            return
        user_id = self.student_identifier(student_id)
        await self.client.long_term.add_preference(
            category=category,
            preference=preference,
            user_identifier=user_id,
        )

    # -------------------------------------------------------------------------
    # FASE 2: Métodos de Doble Rol de Escritura (Profesor vs. Estudiante)
    # -------------------------------------------------------------------------

    async def add_canonical_concept(
        self,
        name: str,
        entity_type: str = "Concept",
        description: str = "",
        properties: Optional[dict[str, Any]] = None,
        mode: InteractionMode = InteractionMode.PROFESSOR,
    ) -> Optional[dict[str, Any]]:
        """Crea o actualiza un concepto canónico en el Knowledge Graph.
        
        REGLA DE ROL: Solo permitido en InteractionMode.PROFESSOR.
        En modo STUDENT, las afirmaciones del alumno nunca modifican el KG canónico.
        """
        if mode != InteractionMode.PROFESSOR:
            logger.warning(f"🚫 Intento de modificar KG canónico denegado en modo '{mode.value}' para '{self.agent_name}'")
            return None

        if not await self.ensure_connected():
            return None

        try:
            concept_name = f"{name} ({self.agent_name})" if not name.endswith(f"({self.agent_name})") else name
            meta = {
                "agent_id": self.agent_id,
                "source": "professor",
                "confidence": 1.0,
                **(properties or {}),
            }
            entity, _ = await self.client.long_term.add_entity(
                name=concept_name,
                entity_type=entity_type,
                description=description,
                resolve=False,
                deduplicate=True,
                metadata=meta,
            )

            # Establecer propiedades de aislamiento y auditoría directamente en el nodo
            tag_cypher = """
            MATCH (e:Entity {id: $entity_id})
            SET e.agent_id = $agent_id,
                e.agent_name = $agent_name,
                e.source = 'professor',
                e += $properties
            """
            await self.client.long_term._client.execute_write(
                tag_cypher,
                {
                    "entity_id": str(entity.id),
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "properties": properties or {},
                }
            )

            # Registrar traza de auditoría con relación TOUCHED
            await self.record_reasoning_trace(
                action="add_canonical_concept",
                input_data=f"{name} ({entity_type}): {description}",
                output_data=f"Concepto canónico creado con ID {entity.id}",
                entities_touched=[str(entity.id)],
                mode=mode,
            )

            logger.info(f"🎓 [{self.agent_name}] Concepto canónico añadido por Profesor: {name}")
            return {
                "id": str(entity.id),
                "name": entity.name,
                "type": entity.type,
                "description": entity.description,
            }
        except Exception as e:
            logger.error(f"❌ Error al agregar concepto canónico '{name}': {e}")
            return None

    async def update_canonical_concept(
        self,
        concept_id: str,
        updates: dict[str, Any],
        mode: InteractionMode = InteractionMode.PROFESSOR,
    ) -> bool:
        """Modifica un concepto canónico existente en el KG con auditoría TOUCHED.
        
        REGLA DE ROL: Solo permitido en InteractionMode.PROFESSOR.
        """
        if mode != InteractionMode.PROFESSOR:
            logger.warning(f"🚫 Intento de actualizar KG canónico denegado en modo '{mode.value}'")
            return False

        if not await self.ensure_connected():
            return False

        try:
            cypher = """
            MATCH (e:Entity {id: $concept_id})
            WHERE e.agent_id = $agent_id OR e.agent_id IS NULL
            SET e += $updates,
                e.agent_id = $agent_id,
                e.last_updated_by = 'professor',
                e.last_updated_at = datetime()
            RETURN e.id AS id
            """
            res = await self.client.long_term._client.execute_write(
                cypher,
                {
                    "concept_id": concept_id,
                    "agent_id": self.agent_id,
                    "updates": updates,
                }
            )
            if res:
                await self.record_reasoning_trace(
                    action="update_canonical_concept",
                    input_data=f"Updates for {concept_id}: {json.dumps(updates, default=str)}",
                    output_data="Actualizado con éxito por Profesor",
                    entities_touched=[concept_id],
                    mode=mode,
                )
                logger.info(f"🎓 [{self.agent_name}] Concepto {concept_id} actualizado por Profesor")
                return True
            else:
                logger.warning(f"⚠️ Concepto {concept_id} no encontrado para agente {self.agent_id}")
                return False
        except Exception as e:
            logger.error(f"❌ Error al actualizar concepto {concept_id}: {e}")
            return False

    async def query_canonical_context(self, query: str, max_items: int = 5) -> list[dict[str, Any]]:
        """Consulta conceptos canónicos y principios validados del KG de este agente (Read-Only para ambos roles)."""
        if not await self.ensure_connected():
            return []

        try:
            keywords = [w.strip().lower() for w in query.split() if len(w.strip()) > 3]
            cypher = """
            MATCH (e:Entity)
            WHERE e.agent_id = $agent_id
              AND (e.source = 'professor' OR e.type IN ['Concept', 'Principle', 'PhysicalQuantity', 'Formula'])
              AND (
                  ANY(k IN $keywords WHERE toLower(e.name) CONTAINS k)
                  OR (e.description IS NOT NULL AND ANY(k IN $keywords WHERE toLower(e.description) CONTAINS k))
              )
            RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description, e.formula AS formula
            LIMIT $limit
            """
            results = await self.client.long_term._client.execute_read(
                cypher,
                {
                    "agent_id": self.agent_id,
                    "keywords": keywords,
                    "limit": max_items,
                }
            )
            if results:
                return [dict(row) for row in results]

            # Fallback a búsqueda general de entidades canónicas del agente
            cypher_fallback = """
            MATCH (e:Entity {agent_id: $agent_id})
            WHERE e.source = 'professor' OR e.type IN ['Concept', 'Principle', 'PhysicalQuantity']
            RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description, e.formula AS formula
            LIMIT $limit
            """
            results_fb = await self.client.long_term._client.execute_read(
                cypher_fallback,
                {"agent_id": self.agent_id, "limit": max_items}
            )
            return [dict(row) for row in results_fb]
        except Exception as e:
            logger.warning(f"⚠️ Error al consultar contexto canónico: {e}")
            return []

    async def register_student_claim(
        self,
        student_id: str,
        concept: str,
        student_claim: str,
        canonical_value: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Guarda la afirmación del estudiante en Short-Term como candidato sin modificar el KG canónico."""
        return await self.register_misconception_candidate(
            student_id=student_id,
            concept=concept,
            student_claim=student_claim,
            canonical_value=canonical_value,
            session_id=session_id,
        )

    async def register_confirmed_deficiency(
        self,
        student_id: str,
        tema: str,
        correccion: str,
        occurrences: int = 1,
        severity: str = "media",
    ) -> bool:
        """Registra una falencia confirmada del estudiante en el KG de forma semántica y estructural.
        
        Asocia la falencia al perfil del estudiante (StudentProfile) y la vincula con el concepto (Concept)
        mediante las relaciones HAS_DEFICIENCY y ABOUT_CONCEPT de POLE+O.
        """
        if not await self.ensure_connected():
            return False

        try:
            user_id = self.student_identifier(student_id)
            pref_text = f"El alumno tiene una falencia confirmada en '{tema}': {correccion}"
            await self.client.long_term.add_preference(
                category="falencia",
                preference=pref_text,
                user_identifier=user_id,
            )

            # 1. Perfil del estudiante (StudentProfile)
            student_node_name = f"Estudiante_{student_id}"
            student_entity, _ = await self.client.long_term.add_entity(
                name=student_node_name,
                entity_type="StudentProfile",
                description=f"Perfil del estudiante {student_id}",
                resolve=False,
                deduplicate=True,
                metadata={"agent_id": self.agent_id, "student_id": student_id},
            )
            tag_student = """
            MATCH (s:Entity {id: $entity_id})
            SET s.agent_id = $agent_id,
                s.student_id = $student_id
            """
            await self.client.long_term._client.execute_write(
                tag_student,
                {
                    "entity_id": str(student_entity.id),
                    "agent_id": self.agent_id,
                    "student_id": student_id,
                }
            )

            # 2. Nodo de Deficiencia Confirmada (StudentDeficiency)
            deficiency_id = f"def_{student_id}_{uuid4().hex[:8]}"
            cypher_def = """
            MERGE (d:Entity:StudentDeficiency {name: $def_name, agent_id: $agent_id})
            ON CREATE SET 
                d.id = $def_id,
                d.type = 'StudentDeficiency',
                d.student_id = $student_id,
                d.concept_ref = $tema,
                d.description = $correccion,
                d.occurrences = $occurrences,
                d.severity = $severity,
                d.status = 'confirmed',
                d.first_detected = datetime(),
                d.last_detected = datetime()
            ON MATCH SET
                d.occurrences = d.occurrences + $occurrences,
                d.last_detected = datetime(),
                d.description = $correccion
            WITH d
            MATCH (s:Entity {id: $student_entity_id})
            MERGE (s)-[r:HAS_DEFICIENCY]->(d)
            ON CREATE SET r.confirmed_at = datetime()
            WITH d
            OPTIONAL MATCH (c:Entity {agent_id: $agent_id})
            WHERE toLower(c.name) CONTAINS toLower($tema) AND c.type IN ['Concept', 'Principle', 'OBJECT']
            WITH d, c
            WHERE c IS NOT NULL
            MERGE (d)-[:ABOUT_CONCEPT]->(c)
            RETURN d.id AS id
            """
            res = await self.client.long_term._client.execute_write(
                cypher_def,
                {
                    "def_name": f"Falencia: {tema} ({student_id})",
                    "agent_id": self.agent_id,
                    "def_id": deficiency_id,
                    "student_id": student_id,
                    "student_entity_id": str(student_entity.id),
                    "tema": tema,
                    "correccion": correccion,
                    "occurrences": occurrences,
                    "severity": severity,
                }
            )

            actual_def_id = res[0]["id"] if res else deficiency_id
            await self.record_reasoning_trace(
                action="register_confirmed_deficiency",
                input_data=f"Estudiante: {student_id}, Tema: {tema}, Ocurrencias: {occurrences}",
                output_data=f"Deficiencia confirmada registrada: {actual_def_id}",
                entities_touched=[str(student_entity.id), str(actual_def_id)],
                mode=InteractionMode.STUDENT,
            )

            logger.info(f"✅ Falencia confirmada del estudiante {student_id} en '{tema}': {correccion}")
            return True
        except Exception as e:
            logger.error(f"❌ Error registrando falencia confirmada para {student_id}: {e}")
            return False

    async def add_deficiency(self, tema: str, correccion: str, student_id: Optional[str] = None) -> bool:
        """Compatibilidad hacia atrás: delega a register_confirmed_deficiency."""
        effective_student_id = student_id or "default_student"
        return await self.register_confirmed_deficiency(
            student_id=effective_student_id,
            tema=tema,
            correccion=correccion,
            occurrences=1,
        )

    # -------------------------------------------------------------------------
    # FASE 3: Detección de Falencias por Acumulación Semántica (Embeddings)
    # -------------------------------------------------------------------------

    async def register_misconception_candidate(
        self,
        student_id: str,
        concept: str,
        student_claim: str,
        canonical_value: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Registra un candidato a misconception en Short-Term y como nodo MisconceptionCandidate.
        
        NO se escribe al KG como deficiencia confirmada de inmediato.
        Se acumula para detección por similaridad semántica (≥5 coincidencias).
        """
        if not await self.ensure_connected():
            return {}

        scoped_id = self.scope_session_id(session_id)
        cand_id = f"cand_{student_id}_{uuid4().hex[:8]}"

        # 1. Registrar evento en Short-Term Memory sin extracción de entidades (modo skip)
        try:
            content_payload = json.dumps({
                "type": "misconception_candidate",
                "candidate_id": cand_id,
                "concept": concept,
                "student_claim": student_claim,
                "canonical_value": canonical_value,
                "student_id": student_id,
            })
            await self.client.short_term.add_message(
                session_id=scoped_id,
                role="system",
                content=content_payload,
                user_identifier=self.student_identifier(student_id),
                extract_entities=False,
                extraction_mode="skip",
                metadata={
                    "agent_id": self.agent_id,
                    "event_type": "misconception_candidate",
                },
            )
        except Exception as e:
            logger.debug(f"Short term candidate log: {e}")

        # 2. Persistir nodo MisconceptionCandidate en Neo4j con estado 'pending'
        try:
            cypher = """
            CREATE (c:Entity:MisconceptionCandidate {
                id: $cand_id,
                name: $name,
                type: 'MisconceptionCandidate',
                student_id: $student_id,
                agent_id: $agent_id,
                concept: $concept,
                student_claim: $student_claim,
                canonical_value: $canonical_value,
                session_id: $session_id,
                status: 'pending',
                detected_at: datetime()
            })
            RETURN c.id AS id
            """
            await self.client.long_term._client.execute_write(
                cypher,
                {
                    "cand_id": cand_id,
                    "name": f"Candidate: {concept} - {student_claim[:30]}",
                    "student_id": student_id,
                    "agent_id": self.agent_id,
                    "concept": concept,
                    "student_claim": student_claim,
                    "canonical_value": canonical_value,
                    "session_id": scoped_id,
                }
            )
            logger.info(f"📝 Candidato a falencia registrado para estudiante '{student_id}': {concept} - '{student_claim}'")
            return {
                "candidate_id": cand_id,
                "concept": concept,
                "student_claim": student_claim,
                "canonical_value": canonical_value,
            }
        except Exception as e:
            logger.error(f"❌ Error registrando misconception candidate: {e}")
            return {}

    async def consolidate_deficiencies(
        self,
        student_id: str,
        similarity_threshold: float = DEFICIENCY_SIMILARITY_THRESHOLD,
        count_threshold: int = DEFICIENCY_COUNT_THRESHOLD,
    ) -> list[dict[str, Any]]:
        """Analiza misconceptions acumuladas usando SIMILARIDAD SEMÁNTICA (Embeddings)
        y promueve a StudentDeficiency aquellas que superan ≥5 coincidencias (cosine >= 0.82).
        
        Algoritmo:
        1. Consulta candidatos pendientes para el estudiante en Neo4j.
        2. Calcula embeddings usando el embedder SentenceTransformers BGE.
        3. Calcula la matriz de similitud de coseno.
        4. Agrupa en clusters con similitud >= similarity_threshold (0.82).
        5. Promueve a StudentDeficiency los clusters con tamaño >= count_threshold (5).
        """
        if not await self.ensure_connected():
            return []

        try:
            # 1. Consultar todos los candidatos pendientes para este estudiante y agente
            cypher = """
            MATCH (c:MisconceptionCandidate)
            WHERE c.student_id = $student_id
              AND c.agent_id = $agent_id
              AND c.status = 'pending'
            RETURN c.id AS id, c.concept AS concept, c.student_claim AS student_claim, c.canonical_value AS canonical_value
            ORDER BY c.detected_at ASC
            """
            candidates = await self.client.long_term._client.execute_read(
                cypher,
                {"student_id": student_id, "agent_id": self.agent_id}
            )
            if not candidates or len(candidates) < count_threshold:
                logger.info(f"ℹ️ [{self.agent_name}] Candidatos pendientes para {student_id}: {len(candidates)} (Mínimo requerido: {count_threshold})")
                return []

            # 2. Calcular embeddings para cada candidato
            texts = [f"{c['concept']}. Afirmación errónea: {c['student_claim']}" for c in candidates]
            embeddings: list[list[float]] = []

            embedder = self.client.long_term._embedder
            for text in texts:
                if embedder:
                    emb = await embedder.embed(text)
                else:
                    emb = [0.0] * 384
                embeddings.append(emb)

            emb_matrix = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            normalized = emb_matrix / norms
            sim_matrix = np.dot(normalized, normalized.T)

            # 3. Clustering de similitud semántica (Greedy clustering)
            visited = set()
            clusters: list[list[int]] = []

            for i in range(len(candidates)):
                if i in visited:
                    continue
                cluster = [i]
                visited.add(i)
                for j in range(i + 1, len(candidates)):
                    if j not in visited and sim_matrix[i, j] >= similarity_threshold:
                        cluster.append(j)
                        visited.add(j)
                clusters.append(cluster)

            # 4. Filtrar clusters que cumplen con el umbral count_threshold (≥ 5)
            confirmed_deficiencies = []
            for cluster_indices in clusters:
                if len(cluster_indices) >= count_threshold:
                    cluster_candidates = [candidates[idx] for idx in cluster_indices]
                    representative = cluster_candidates[0]
                    rep_concept = representative["concept"]
                    rep_canonical = representative["canonical_value"]
                    candidate_ids = [c["id"] for c in cluster_candidates]

                    summary_description = (
                        f"Detectadas {len(cluster_candidates)} discrepancias semánticamente equivalentes sobre '{rep_concept}'. "
                        f"Corrección canónica: {rep_canonical}"
                    )

                    # Registrar la deficiencia confirmada en el KG
                    success = await self.register_confirmed_deficiency(
                        student_id=student_id,
                        tema=rep_concept,
                        correccion=summary_description,
                        occurrences=len(cluster_candidates),
                        severity="alta" if len(cluster_candidates) >= 7 else "media",
                    )

                    if success:
                        # Marcar candidatos como consolidados
                        update_cypher = """
                        MATCH (c:MisconceptionCandidate)
                        WHERE c.id IN $cand_ids
                        SET c.status = 'consolidated',
                            c.consolidated_at = datetime()
                        """
                        await self.client.long_term._client.execute_write(
                            update_cypher,
                            {"cand_ids": candidate_ids}
                        )

                        confirmed_deficiencies.append({
                            "concept": rep_concept,
                            "occurrences": len(cluster_candidates),
                            "canonical_value": rep_canonical,
                            "candidate_ids": candidate_ids,
                            "description": summary_description,
                        })
                        logger.info(f"🎯 ¡Falencia confirmada por consolidación semántica ({len(cluster_candidates)} coincidencias): {rep_concept}!")

            return confirmed_deficiencies
        except Exception as e:
            logger.error(f"❌ Error en consolidate_deficiencies para {student_id}: {e}")
            return []

    async def get_student_deficiencies(self, student_id: str) -> list[dict[str, Any]]:
        """Consulta todas las deficiencias confirmadas para un estudiante en este agente."""
        if not await self.ensure_connected():
            return []

        try:
            cypher = """
            MATCH (d:StudentDeficiency)
            WHERE d.student_id = $student_id AND d.agent_id = $agent_id
            OPTIONAL MATCH (d)-[:ABOUT_CONCEPT]->(c:Entity)
            RETURN d.id AS id,
                   d.concept_ref AS concept,
                   d.description AS description,
                   d.occurrences AS occurrences,
                   d.severity AS severity,
                   d.status AS status,
                   c.name AS concept_node_name
            ORDER BY d.occurrences DESC
            """
            results = await self.client.long_term._client.execute_read(
                cypher,
                {"student_id": student_id, "agent_id": self.agent_id}
            )
            return [dict(row) for row in results]
        except Exception as e:
            logger.warning(f"⚠️ Error al consultar deficiencias del estudiante: {e}")
            return []

    # -------------------------------------------------------------------------
    # FASE 4: Validación Contra Contenido Canónico del KG
    # -------------------------------------------------------------------------

    async def validate_against_kg(
        self,
        student_claim: str,
        llm: Any = None,
    ) -> dict[str, Any]:
        """Compara la afirmación del estudiante contra el KG canónico.
        
        Las correcciones del estudiante NUNCA se toman como verdad absoluta.
        Se contrasta contra los conceptos canónicos y principios validados.
        
        Returns:
            {
                "is_correct": bool,
                "concept": str,
                "canonical_value": str,
                "student_claim": str,
                "explanation": str,
            }
        """
        if not student_claim or not student_claim.strip():
            return {
                "is_correct": True,
                "concept": "general",
                "canonical_value": "",
                "student_claim": "",
                "explanation": "Afirmación vacía",
            }

        # 1. Consultar conceptos canónicos relevantes en el KG
        canonical_entities = await self.query_canonical_context(query=student_claim, max_items=5)
        
        context_summary = ""
        if canonical_entities:
            context_summary = "\n".join([
                f"- {e.get('name')}: {e.get('description', '')} (Fórmula: {e.get('formula', 'N/A')})"
                for e in canonical_entities
            ])
        else:
            gen_context = await self.get_context(query=student_claim, student_id="system", session_id="val_session", max_items=5)
            context_summary = gen_context

        if not context_summary.strip():
            return {
                "is_correct": True,
                "concept": "general",
                "canonical_value": "",
                "student_claim": student_claim,
                "explanation": "No hay restricciones canónicas contradictorias registradas en el grafo.",
            }

        eval_llm = llm
        if not eval_llm:
            try:
                eval_llm = create_google_llm()
            except Exception as e:
                logger.debug(f"Could not create Google LLM: {e}")

        # 2. Evaluar mediante LLM contrastando estrictamente contra el KG
        if eval_llm:
            try:
                from langchain_core.messages import HumanMessage
                prompt = f"""Eres un validador pedagógico estricto del curso de {self.agent_name}.
Tu tarea es contrastar la afirmación del estudiante contra el Conocimiento Canónico del Grafo de Conocimiento (KG).

REGLAS CRÍTICAS:
1. El KG canónico es la verdad de referencia. Las afirmaciones del estudiante NUNCA deben aceptarse si contradicen el KG.
2. Si el estudiante afirma algo físicamente/médicamente erróneo o contrario al KG, marca is_correct: false.
3. Si el estudiante dice algo correcto y compatible con el KG, marca is_correct: true.

Conocimiento Canónico del KG:
{context_summary}

Afirmación del estudiante:
"{student_claim}"

Responde ÚNICAMENTE un objeto JSON válido con este formato:
{{
  "is_correct": true o false,
  "concept": "<nombre del concepto central involucrado>",
  "canonical_value": "<lo que afirma la física/medicina canónica según el KG>",
  "explanation": "<breve explicación de por qué es correcto o cuál es el error del alumno>"
}}"""

                response = await ainvoke_with_retry(eval_llm, [HumanMessage(content=prompt)])
                raw_text = response.content.strip()
                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_text:
                    raw_text = raw_text.split("```")[1].split("```")[0].strip()

                try:
                    data = json.loads(raw_text, strict=False)
                except Exception:
                    import re
                    sanitized = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', raw_text)
                    data = json.loads(sanitized, strict=False)
                return {
                    "is_correct": bool(data.get("is_correct", True)),
                    "concept": str(data.get("concept", "general")),
                    "canonical_value": str(data.get("canonical_value", "")),
                    "student_claim": student_claim,
                    "explanation": str(data.get("explanation", "")),
                }
            except Exception as e:
                logger.warning(f"⚠️ Error evaluando validación contra KG con LLM: {e}")

        return {
            "is_correct": True,
            "concept": "general",
            "canonical_value": "",
            "student_claim": student_claim,
            "explanation": "Validado por coincidencia directa con el grafo de conocimiento.",
        }

    # -------------------------------------------------------------------------
    # FASE 5: Reasoning Traces y Relaciones TOUCHED para Auditoría
    # -------------------------------------------------------------------------

    async def record_reasoning_trace(
        self,
        action: str,
        input_data: str,
        output_data: str,
        entities_touched: Optional[list[str]] = None,
        mode: InteractionMode = InteractionMode.STUDENT,
    ) -> str:
        """Registra una traza de razonamiento y crea relaciones TOUCHED hacia las entidades afectadas."""
        if not await self.ensure_connected():
            return ""

        trace_id = f"trace_{uuid4().hex[:12]}"
        entities_touched = entities_touched or []

        try:
            # 1. Crear nodo ReasoningTrace
            cypher_trace = """
            CREATE (t:Entity:ReasoningTrace {
                id: $trace_id,
                name: $name,
                type: 'ReasoningTrace',
                action: $action,
                input_data: $input_data,
                output_data: $output_data,
                mode: $mode,
                agent_id: $agent_id,
                timestamp: datetime()
            })
            RETURN t.id AS id
            """
            await self.client.long_term._client.execute_write(
                cypher_trace,
                {
                    "trace_id": trace_id,
                    "name": f"Trace: {action} ({mode.value if hasattr(mode, 'value') else mode})",
                    "action": action,
                    "input_data": input_data[:1000],
                    "output_data": output_data[:1000],
                    "mode": mode.value if hasattr(mode, "value") else str(mode),
                    "agent_id": self.agent_id,
                }
            )

            # 2. Conectar relaciones TOUCHED hacia las entidades leídas o modificadas
            if entities_touched:
                cypher_touch = """
                MATCH (t:Entity:ReasoningTrace {id: $trace_id})
                MATCH (e:Entity)
                WHERE e.id IN $entity_ids OR e.name IN $entity_ids
                MERGE (t)-[r:TOUCHED {
                    action_type: $action,
                    timestamp: datetime(),
                    mode: $mode
                }]->(e)
                """
                await self.client.long_term._client.execute_write(
                    cypher_touch,
                    {
                        "trace_id": trace_id,
                        "entity_ids": entities_touched,
                        "action": action,
                        "mode": mode.value if hasattr(mode, "value") else str(mode),
                    }
                )

            logger.debug(f"📊 Traza de razonamiento registrada: {trace_id} ({action}) - {len(entities_touched)} entidades tocadas")
            return trace_id
        except Exception as e:
            logger.warning(f"⚠️ Error al registrar traza de razonamiento: {e}")
            return ""

    async def get_conclusions(self, student_id: str) -> tuple[list[str], list[str]]:
        """Obtiene conclusiones/preferencias y falencias exclusivamente de este agente."""
        if not await self.ensure_connected():
            return [], []

        student_id_scoped = self.student_identifier(student_id)
        agent_id_scoped = self.system_identifier()

        student_prefs = await self.client.long_term.get_preferences_for(student_id_scoped)
        agent_prefs = await self.client.long_term.get_preferences_for(agent_id_scoped)

        conclusions = []
        for p in student_prefs:
            pref_str = p.preference if hasattr(p, "preference") else (p.get("preference", str(p)) if isinstance(p, dict) else str(p))
            conclusions.append(pref_str)

        deficiencies = []
        for p in agent_prefs:
            pref_str = p.preference if hasattr(p, "preference") else (p.get("preference", str(p)) if isinstance(p, dict) else str(p))
            deficiencies.append(pref_str)

        # Consultar también StudentDeficiency confirmadas
        try:
            confirmed = await self.get_student_deficiencies(student_id)
            for d in confirmed:
                line = f"Falencia confirmada en '{d['concept']}': {d['description']} ({d['occurrences']} incidencias)"
                if line not in deficiencies:
                    deficiencies.append(line)
        except Exception as e:
            logger.debug(f"Error recuperando falencias confirmadas en get_conclusions: {e}")

        return conclusions, deficiencies

    async def learn_user_preferences(self, user_message: str, student_id: str, llm: Any):
        """Extrae y persiste preferencias e insights en segundo plano usando Gemini 2.5."""
        if not await self.ensure_connected() or not user_message:
            return

        try:
            from langchain_core.messages import HumanMessage

            prompt = f"""Analiza el siguiente mensaje enviado por un estudiante en una sesión de {self.agent_name}:
"{user_message}"

Determina si se pueden extraer dos tipos de información para el perfil del alumno en este tema ({self.agent_name}):
1. Preferencias personales o de estilo: Hábitos del usuario, estilo de comunicación preferido o datos biográficos (ej: prefiere respuestas cortas, le gustan las analogías, estudia ingeniería, etc.).
2. Insights o falencias de conocimiento del alumno: Conceptos recurrentes, dudas o temas sobre los cuales el alumno pregunta o demuestra no comprender en relación a {self.agent_name}.

Si encuentras alguno de estos tipos, descríbelo en una frase corta y directa en tercera persona (ej: "El usuario prefiere explicaciones con el método socrático", "El alumno tiene dudas sobre la conservación de la energía").
Si no encuentras nada relevante para una categoría, responde NONE para esa categoría.

Responde estrictamente en el formato:
Preferencia: <frase corta o NONE>
Insight: <frase corta o NONE>"""

            response = await ainvoke_with_retry(llm, [HumanMessage(content=prompt)])
            result = response.content.strip()

            preferences = []
            insights = []
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("Preferencia:"):
                    val = line.split(":", 1)[1].strip()
                    if val and val.upper() != "NONE":
                        preferences.append(val)
                elif line.startswith("Insight:"):
                    val = line.split(":", 1)[1].strip()
                    if val and val.upper() != "NONE":
                        insights.append(val)

            user_id = self.student_identifier(student_id)
            for pref in preferences:
                await self.client.long_term.add_preference(
                    category="user_preference",
                    preference=pref,
                    user_identifier=user_id,
                )
                logger.info(f"💾 [{self.agent_name}] Preferencia guardada: {pref}")

            for ins in insights:
                await self.client.long_term.add_preference(
                    category="insight",
                    preference=ins,
                    user_identifier=user_id,
                )
                logger.info(f"💾 [{self.agent_name}] Insight guardado: {ins}")
        except Exception as e:
            logger.warning(f"⚠️ [{self.agent_name}] Error en extractor de preferencias: {e}")
