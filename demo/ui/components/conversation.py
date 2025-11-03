# demo/ui/components/conversation.py (FIXED)

import uuid

import mesop as me
from a2a.types import Message, Part, Role, TextPart
from state.host_agent_service import (ListConversations, SendMessage,
                                      convert_message_to_state)
from state.state import AppState, StateMessage

from .chat_bubble import chat_bubble
from .form_render import form_sent, is_form, render_form


@me.stateclass
class PageState:
    """Local Page State"""

    conversation_id: str = ''
    message_content: str = ''


def on_blur(e: me.InputBlurEvent):
    """Input handler"""
    state = me.state(PageState)
    state.message_content = e.value


async def send_message(message: str, message_id: str = ''):
    state = me.state(PageState)
    app_state = me.state(AppState)
    c = next(
        (
            x
            for x in await ListConversations()
            if x.conversation_id == state.conversation_id
        ),
        None,
    )
    if not c:
        print('Conversation id ', state.conversation_id, ' not found')
    request = Message(
        message_id=message_id,
        context_id=state.conversation_id,
        role=Role.user,
        parts=[Part(root=TextPart(text=message))],
    )
    # Add message to state until refresh replaces it.
    state_message = convert_message_to_state(request)
    if not app_state.messages:
        app_state.messages = []
    app_state.messages.append(state_message)
    conversation = next(
        filter(
            lambda x: c and x.conversation_id == c.conversation_id,
            app_state.conversations,
        ),
        None,
    )
    if conversation:
        conversation.message_ids.append(state_message.message_id)
    await SendMessage(request)


async def send_message_enter(e: me.InputEnterEvent):
    """Send message handler"""
    yield
    state = me.state(PageState)
    state.message_content = e.value
    app_state = me.state(AppState)
    
    # CRÍTICO: Generar message_id ANTES de agregarlo a background_tasks
    message_id = str(uuid.uuid4())
    
    # Agregar a background_tasks con mensaje inicial
    app_state.background_tasks[message_id] = 'Enviando mensaje...'
    print(f"✅ Mensaje agregado a background_tasks: {message_id}")
    
    yield
    
    try:
        await send_message(state.message_content, message_id)
        print(f"✅ Mensaje enviado exitosamente: {message_id}")
        
        # CRÍTICO: Limpiar background_tasks después de enviar
        # El polling se encargará de actualizar el estado
        # pero necesitamos remover este ID después de un delay
        # para permitir que el servidor lo procese
        
    except Exception as e:
        print(f"❌ Error enviando mensaje: {e}")
        # Limpiar en caso de error
        if message_id in app_state.background_tasks:
            del app_state.background_tasks[message_id]
    
    # Limpiar el input
    state.message_content = ''
    
    yield


async def send_message_button(e: me.ClickEvent):
    """Send message button handler"""
    yield
    state = me.state(PageState)
    app_state = me.state(AppState)
    
    # CRÍTICO: Generar message_id ANTES
    message_id = str(uuid.uuid4())
    
    # Agregar a background_tasks
    app_state.background_tasks[message_id] = 'Enviando mensaje...'
    print(f"✅ Mensaje agregado a background_tasks: {message_id}")
    
    yield
    
    try:
        await send_message(state.message_content, message_id)
        print(f"✅ Mensaje enviado exitosamente: {message_id}")
        
    except Exception as e:
        print(f"❌ Error enviando mensaje: {e}")
        # Limpiar en caso de error
        if message_id in app_state.background_tasks:
            del app_state.background_tasks[message_id]
    
    # Limpiar el input
    state.message_content = ''
    
    yield


@me.component
def conversation():
    """Conversation component"""
    page_state = me.state(PageState)
    app_state = me.state(AppState)
    if 'conversation_id' in me.query_params:
        page_state.conversation_id = me.query_params['conversation_id']
        app_state.current_conversation_id = page_state.conversation_id
    
    with me.box(
        style=me.Style(
            display='flex',
            justify_content='space-between',
            flex_direction='column',
        )
    ):
        for message in app_state.messages:
            if is_form(message):
                render_form(message, app_state)
            elif form_sent(message, app_state):
                chat_bubble(
                    StateMessage(
                        message_id=message.message_id,
                        role=message.role,
                        content=[('Form submitted', 'text/plain')],
                    ),
                    message.message_id,
                )
            else:
                chat_bubble(message, message.message_id)

        with me.box(
            style=me.Style(
                display='flex',
                flex_direction='row',
                gap=5,
                align_items='center',
                min_width=500,
                width='100%',
            )
        ):
            me.input(
                label='How can I help you?',
                value=page_state.message_content,  # AÑADIDO: Bind al state
                on_blur=on_blur,
                on_enter=send_message_enter,
                style=me.Style(min_width='80vw'),
            )
            with me.content_button(
                type='flat',
                on_click=send_message_button,
            ):
                me.icon(icon='send')
