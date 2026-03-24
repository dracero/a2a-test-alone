"""Utilidades para manejo de mensajes."""

from a2a.types import Message


def get_message_id(m: Message | None) -> str | None:
    """Obtiene el message_id de un mensaje A2A."""
    if not m or not m.metadata or 'message_id' not in m.metadata:
        return None
    return m.metadata['message_id']
