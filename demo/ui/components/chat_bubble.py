import re
from markdown_it import MarkdownIt

import mesop as me

from state.state import AppState, StateMessage

import uuid

# Inicializar convertidor de Markdown
md = MarkdownIt("js-default")

def _render_full_html(content: str, role: str) -> str:
    """Convierte Markdown a HTML protegiendo todos los tags HTML (especialmente <img>)."""
    try:
        # Regex más permisivo para cualquier tag <img>
        img_pattern = r'<img[^>]+>'
        placeholders = {}
        
        def _to_placeholder(match):
            placeholder_id = f"__IMG_{uuid.uuid4().hex}__"
            placeholders[placeholder_id] = match.group(0)
            return placeholder_id
        
        # 1. Reemplazar tags por placeholders
        text_with_placeholders = re.sub(img_pattern, _to_placeholder, content)
        
        # 2. Renderizar Markdown
        html_body = md.render(text_with_placeholders)
        
        # 3. Restaurar los tags originales
        for pid, tag in placeholders.items():
            html_body = html_body.replace(pid, tag)
            
        print(f"DEBUG: [chat_bubble] Rendered {role} ({len(content)} chars). Found {len(placeholders)} images.")
    except Exception as e:
        print(f"⚠️ DEBUG: [chat_bubble] Error: {e}")
        html_body = content.replace('<', '&lt;').replace('>', '&gt;')
    
    return f'''<!DOCTYPE html>
<html>
<head>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 10px 15px;
    line-height: 1.5;
    background: transparent;
    font-size: 15px;
    color: #3c4043;
    word-wrap: break-word;
  }}
  img {{
    max-width: 100%;
    height: auto;
    vertical-align: middle;
    display: inline-block;
  }}
  p {{ margin: 0 0 12px 0; }}
  p:last-child {{ margin-bottom: 0; }}
  strong {{ font-weight: 600; color: #1a73e8; }}
</style>
</head>
<body>
{html_body}
<script>
  function updateHeight() {{
    var h = document.body.scrollHeight;
    window.parent.postMessage({{type: 'resize', height: h}}, '*');
  }}
  window.addEventListener('load', updateHeight);
  setTimeout(updateHeight, 100);
  setTimeout(updateHeight, 500);
  setInterval(updateHeight, 1000);
</script>
</body>
</html>'''


@me.component
def chat_bubble(message: StateMessage, key: str):
    """Chat bubble component"""
    app_state = me.state(AppState)
    show_progress_bar = (
        message.message_id in app_state.background_tasks
        or message.message_id in app_state.message_aliases.values()
    )
    progress_text = ''
    if show_progress_bar:
        progress_text = app_state.background_tasks[message.message_id]
    
    for i, pair in enumerate(message.content):
        chat_box(
            pair[0],
            pair[1],
            message.role,
            f"{key}_{i}", # Usar un ID único por parte
            progress_bar=show_progress_bar,
            progress_text=progress_text,
        )


def chat_box(
    content: str,
    media_type: str,
    role: str,
    key: str,
    progress_bar: bool,
    progress_text: str,
):
    # Detección de rol robusta
    is_user = role.lower() in ('user', 'human')
    
    with me.box(
        style=me.Style(
            display='flex',
            justify_content=('flex-end' if is_user else 'flex-start'),
            min_width='100%',
        ),
        key=key,
    ):
        with me.box(
            style=me.Style(
                display='flex', 
                flex_direction='column', 
                max_width='85%',
                gap=5
            )
        ):
            if media_type == 'image/png' and '<img' not in content:
                # Imagen real (no fórmula)
                img_src = content
                if '/message/file' not in content:
                    img_src = 'data:image/png;base64,' + content
                me.image(
                    src=img_src,
                    style=me.Style(
                        width='100%',
                        max_width=500,
                        border_radius=10,
                        object_fit='contain',
                    ),
                )
            elif not is_user:
                # Agent: render with HTML/placeholder strategy
                full_html = _render_full_html(content, role)
                with me.box(
                    style=me.Style(
                        box_shadow='0 1px 2px 0 rgba(60,64,67,0.3)',
                        background=me.theme_var('secondary-container'),
                        border_radius=15,
                        overflow_x='hidden',
                        margin=me.Margin(top=5, bottom=5),
                    ),
                ):
                    me.html(
                        full_html,
                        style=me.Style(
                            width='100%',
                            min_height=40,
                        ),
                    )
            else:
                # User: standard Markdown
                me.markdown(
                    content,
                    style=me.Style(
                        font_family='Google Sans',
                        box_shadow='0 1px 2px 0 rgba(60,64,67,0.3)',
                        padding=me.Padding(top=5, left=15, right=15, bottom=5),
                        margin=me.Margin(top=5, bottom=5),
                        background=me.theme_var('primary-container'),
                        border_radius=15,
                    ),
                )
    
    if progress_bar:
        with me.box(
            style=me.Style(
                display='flex',
                justify_content=('flex-start' if role == 'user' else 'flex-end'),
                min_width='100%',
            ),
        ):
            with me.box(
                style=me.Style(
                    display='flex',
                    flex_direction='column',
                    max_width='80%',
                    gap=5
                )
            ):
                with me.box(
                    style=me.Style(
                        font_family='Google Sans',
                        box_shadow='0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15)',
                        padding=me.Padding(top=10, left=15, right=15, bottom=10),
                        margin=me.Margin(top=5, bottom=5),
                        background=(
                            me.theme_var('primary-container')
                            if role == 'user'
                            else me.theme_var('secondary-container')
                        ),
                        border_radius=15,
                    ),
                ):
                    if not progress_text:
                        progress_text = 'Working...'
                    me.text(progress_text)
                    me.progress_bar(color='accent')
