import asyncio
import os
import json
import logging
from typing import Any, Optional
from uuid import uuid4

from pydantic import SecretStr
from neo4j_agent_memory import MemoryClient, MemorySettings, ExtractionConfig, ExtractorType
from neo4j_agent_memory.llm.adapters.sentence_transformers import SentenceTransformersProvider

try:
    from .api_key_rotator import google_key_rotator, ainvoke_with_retry
except ImportError:
    from api_key_rotator import google_key_rotator, ainvoke_with_retry

logger = logging.getLogger(__name__)


class AgentNAMSMemory:
    """Gestor de memoria NAMS independiente y aislado para un agente específico.

    Garantiza que:
    1. Las sesiones a corto plazo estén prefijadas con el identificador del agente (ej. 'physics:session_123').
    2. Las preferencias e insights estén asociados únicamente al par (estudiante, agente).
    3. Las entidades y conceptos del grafo de conocimiento pertenezcan estrictamente al dominio de este agente,
       impidiendo cualquier cruce de contexto entre agentes (ej: histología vs. física).
    4. El extractor de entidades use LiteLLM con Gemini 2.5 Flash y el rotador de API keys.
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ):
        self.agent_id = agent_id.lower().strip()
        self.agent_name = agent_name.strip()
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database or "neo4j"
        self._connected = False
        self.client: Optional[MemoryClient] = None
        self._init_client()

    def _init_client(self):
        """Inicializa la configuración de MemorySettings y el cliente NAMS con Gemini 2.5."""
        if not (self.uri and self.username and self.password):
            logger.warning(f"⚠️ Parámetros de Neo4j incompletos para el agente '{self.agent_name}'. Memoria inactiva.")
            self.client = None
            return

        # Sincronizar clave activa de Google con LiteLLM
        active_key = google_key_rotator.get_key()
        if active_key:
            os.environ["GEMINI_API_KEY"] = active_key
            os.environ["GOOGLE_API_KEY"] = active_key

        try:
            embedder = SentenceTransformersProvider(
                model="BAAI/bge-small-en-v1.5",
                device="cpu",
            )
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

        # Refrescar credenciales en LiteLLM antes de conectar
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

    async def add_user_message(self, session_id: str, content: str, student_id: Optional[str] = None):
        """Guarda un mensaje del usuario en la sesión aislada del agente."""
        if not await self.ensure_connected():
            return
        scoped_id = self.scope_session_id(session_id)
        user_id = self.student_identifier(student_id or session_id)
        try:
            # Asegurar API key en LiteLLM para la extracción
            active_key = google_key_rotator.get_key()
            if active_key:
                os.environ["GEMINI_API_KEY"] = active_key
                os.environ["GOOGLE_API_KEY"] = active_key

            msg = await self.client.short_term.add_message(
                session_id=scoped_id,
                role="user",
                content=content,
                user_identifier=user_id,
                metadata={"agent_id": self.agent_id, "agent_name": self.agent_name},
            )
            # Etiquetar entidades extraídas con el agent_id para filtrado estricto
            await self._tag_entities_for_message(str(msg.id))
            logger.info(f"💾 [{self.agent_name}] Guardado mensaje de usuario en sesión {scoped_id}")
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
            await self._tag_entities_for_message(str(msg.id))
            logger.info(f"💾 [{self.agent_name}] Guardada respuesta de asistente en sesión {scoped_id}")
        except Exception as e:
            logger.warning(f"⚠️ [{self.agent_name}] Error guardando respuesta de asistente: {e}")

    async def _tag_entities_for_message(self, message_id: str):
        """Etiqueta todas las entidades asociadas al mensaje con el agent_id para aislamiento absoluto."""
        try:
            cypher = """
            MATCH (m:Message {id: $message_id})
            OPTIONAL MATCH (m)-[:MENTIONS]->(e1:Entity)
            OPTIONAL MATCH (e2:Entity)-[:EXTRACTED_FROM]->(m)
            WITH coalesce(e1, e2) AS e
            WHERE e IS NOT NULL
            SET e.agent_id = $agent_id,
                e.agent_name = $agent_name
            """
            await self.client.long_term._client.execute_write(
                cypher,
                {
                    "message_id": message_id,
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
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

        # 2. Preferencias e insights del alumno para este agente y falencias de este sistema
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
            # Consulta Cypher que solo trae entidades asociadas a este agente
            agent_entities_query = """
            MATCH (e:Entity)
            WHERE e.agent_id = $agent_id
               OR EXISTS {
                   MATCH (c:Conversation)-[:HAS_MESSAGE]->(m:Message)
                   WHERE c.session_id STARTS WITH $agent_prefix
                     AND ((m)-[:MENTIONS]->(e) OR (e)-[:EXTRACTED_FROM]->(m))
               }
            RETURN DISTINCT e.name AS name, e.type AS type, e.description AS description
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
                    line = f"- {name} ({etype})"
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

    async def add_deficiency(self, tema: str, correccion: str) -> bool:
        """Guarda una falencia del sistema/agente tanto semántica como estructuralmente."""
        if not await self.ensure_connected():
            return False
        try:
            user_id = self.system_identifier()
            pref_text = f"El sistema/agente {self.agent_name} tiene una falencia en '{tema}': {correccion}"
            await self.client.long_term.add_preference(
                category="falencia",
                preference=pref_text,
                user_identifier=user_id,
            )

            # Guardar estructuralmente en el grafo con scope del agente
            agent_entity, _ = await self.client.long_term.add_entity(
                name=self.agent_name,
                entity_type="Agent",
                description=f"Perfil del agente {self.agent_name}",
                resolve=False,
                deduplicate=True,
                metadata={"agent_id": self.agent_id},
            )

            concept_name = f"{tema} ({self.agent_name})"
            concept_entity, _ = await self.client.long_term.add_entity(
                name=concept_name,
                entity_type="Concept",
                description=f"Concepto para el agente {self.agent_name}: {tema}",
                resolve=False,
                deduplicate=True,
                metadata={"agent_id": self.agent_id},
            )

            await self.client.long_term.add_relationship(
                source=agent_entity.id,
                target=concept_entity.id,
                relationship_type="TIENE_FALENCIA",
                description=correccion,
            )
            logger.info(f"✅ Falencia registrada para agente {self.agent_name}: {tema}")
            return True
        except Exception as e:
            logger.error(f"❌ Error al registrar falencia para {self.agent_name}: {e}")
            return False

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
