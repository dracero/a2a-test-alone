"""
BeeAI Workflow-based Orchestrator
Compatible with Groq (Llama 4)
Uses explicit workflow steps instead of ReAct pattern
"""

import json
from typing import Any

from beeai_framework.workflows.workflow import Workflow
from pydantic import BaseModel

from .api_key_rotator import ainvoke_with_retry
from .langsmith_config import traceable


class OrchestratorState(BaseModel):
    """State for the orchestration workflow"""
    user_message: str
    has_images: bool
    image_data_list: list[dict] = []  # Lista de {mime_type, bytes_b64} para clasificación visual
    available_agents: list[dict] = []
    chosen_agent: str = ""
    agent_response: str = ""
    error: str = ""
    history_text: str = ""
    neo4j_context_text: str = ""
    context_id: str = ""
    student_id: str = ""


async def create_orchestrator_workflow(manager, list_tool, send_tool, llm):
    """
    Create a BeeAI Workflow for orchestrating agent selection and delegation.
    This pattern is fully compatible with Gemini as it doesn't rely on tool calling.
    """
    
    # Step 1: List available agents
    async def list_agents(state: OrchestratorState) -> str:
        """List all available remote agents"""
        print("📋 Step 1: Listing available agents...")
        
        try:
            from service.server.beeai_host_manager import ListRemoteAgentsInput
            agents_json = await list_tool._run(ListRemoteAgentsInput(), None, None)
            state.available_agents = json.loads(agents_json)
            
            print(f"✅ Found {len(state.available_agents)} agents:")
            for agent in state.available_agents:
                print(f"   - {agent['name']}: {agent['description']}")
            
            return "classify_and_choose"
        except Exception as e:
            state.error = f"Error listing agents: {str(e)}"
            print(f"❌ {state.error}")
            return None
    
    # Step 2: Use Groq to classify and choose the best agent
    @traceable(name="orchestrator_classify_and_choose", run_type="chain", tags=["agent_type:orchestrator", "orchestrator"])
    async def classify_and_choose(state: OrchestratorState) -> str:
        """Use multimodal LLM to analyze the request (including images) and choose the best agent"""
        print("🤔 Step 2: Classifying request and choosing agent...")
        
        if not state.available_agents:
            state.error = "No agents available"
            return None
        
        try:
            # Build a clear description of available agents
            agents_description = "\n".join([
                f"{i+1}. {agent['name']}: {agent['description']}" 
                for i, agent in enumerate(state.available_agents)
            ])
            
            # Create a classification prompt
            classification_text = (
                f"You are a routing system. Analyze the user's request and decide which specialized agent should handle it, or if you should respond directly.\n\n"
                f"Available specialized agents:\n{agents_description}\n\n"
            )
            
            if state.neo4j_context_text:
                classification_text += f"User preferences (background):\n{state.neo4j_context_text[:1500]}\n\n"
            
            if state.history_text:
                classification_text += f"Recent conversation history:\n{state.history_text}\n\n"
                
            classification_text += f"User request: \"{state.user_message}\"\n\n"
            
            # If images are present, add visual analysis instructions
            if state.has_images and state.image_data_list:
                classification_text += (
                    f"The user has also attached {len(state.image_data_list)} image(s). "
                    f"LOOK AT THE IMAGE(S) CAREFULLY and determine what they contain. "
                    f"Based on the visual content of the images AND the text, choose the right agent.\n\n"
                )
            
            classification_text += (
                f"RULES:\n"
                f"1. If the request is a simple greeting (hello, hi, hola, hey, etc.) or small talk with NO images, respond with: DIRECT\n"
                f"2. If the request is about general capabilities or help, respond with: DIRECT\n"
                f"3. Route the message to a specialized agent based strictly on the semantic domain and intent of the request:\n"
                f"   - **Asistente Médico**: Route ANY request related to medicine, biology, histology, anatomy, tissues, organs, cells, or clinical topics here. This includes requests to search, show, or retrieve microscopic images or figures, as well as medical text questions.\n"
                f"   - **Tutor Socrático de Física Multimodal**: Route any request related to physics, equations, mechanics, or physical science problems here.\n"
                f"   - **Image Generator Agent**: Route requests here ONLY if the user explicitly asks to generate, create, draw, or paint a generic, artistic, creative, or non-medical synthetic image (e.g., 'draw a red cat', 'generate an image of a beach'). Never route medical or microscopic image retrieval/search requests here.\n"
                f"4. Look at the actual content of any attached images to help identify the domain.\n\n"
                f"First, analyze the user request step-by-step to identify their intent and reason about which agent or DIRECT is best. "
                f"Finally, output your final selection enclosed inside <route> and </route> tags. "
                f"Example: <route>Tutor Socrático de Física Multimodal</route> or <route>DIRECT</route>."
            )
            
            # Build the message content - multimodal if images are present
            from langchain_core.messages import HumanMessage
            
            if state.has_images and state.image_data_list:
                # Multimodal classification: send images + text to the LLM
                content = [{"type": "text", "text": classification_text}]
                
                for idx, img in enumerate(state.image_data_list):
                    mime_type = img.get("mime_type", "image/png")
                    bytes_b64 = img.get("bytes_b64", "")
                    if bytes_b64:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{bytes_b64}"
                            }
                        })
                        print(f"🖼️ Image {idx} included for visual classification: {mime_type}")
                
                classification_llm = getattr(manager, 'vision_llm', None) or llm
                print(f"🔍 Sending MULTIMODAL classification prompt to Groq ({len(state.image_data_list)} images) using model: {getattr(classification_llm, 'model', 'unknown')}...")
                try:
                    llm_response = await ainvoke_with_retry(classification_llm, [HumanMessage(content=content)])
                except Exception as vision_err:
                    print(f"⚠️ Vision model failed: {vision_err}")
                    print(f"🔄 Falling back to text-only LLM for classification...")
                    llm_response = await ainvoke_with_retry(llm, [HumanMessage(content=classification_text)])
            else:
                # Text-only classification
                print(f"🔍 Sending text-only classification prompt to Groq...")
                llm_response = await ainvoke_with_retry(llm, [HumanMessage(content=classification_text)])
            
            print(f"📥 Groq response type: {type(llm_response)}")
            print(f"📥 Groq response content: {llm_response.content}")
            
            raw_response = str(llm_response.content).strip() if llm_response.content else ""
            print(f"🎯 Raw chosen: '{raw_response}'")
            
            if not raw_response:
                print(f"⚠️ Groq returned empty response, using first agent")
                state.chosen_agent = state.available_agents[0]['name']
                return "send_to_agent"
            
            # Extract choice from <route> tags
            import re
            match = re.search(r'<route>(.*?)</route>', raw_response, re.DOTALL | re.IGNORECASE)
            if match:
                chosen = match.group(1).strip()
                print(f"🎯 Extracted choice: '{chosen}'")
            else:
                lines = [line.strip() for line in raw_response.split('\n') if line.strip()]
                chosen = lines[-1] if lines else ""
                chosen = chosen.replace('"', '').replace("'", '').strip()
                if chosen and chosen[0].isdigit():
                    parts = chosen.split('.', 1)
                    if len(parts) > 1:
                        chosen = parts[1].strip()
                print(f"🎯 Fallback chosen: '{chosen}'")
            
            # Check if should respond directly
            if chosen.upper() == 'DIRECT':
                print(f"✅ Responding directly (no agent needed)")
                # Generate a direct response
                direct_prompt = (
                    f"You are a friendly AI assistant.\n"
                )
                if state.neo4j_context_text:
                    direct_prompt += (
                        f"Background context (low priority) - User preferences from memory:\n"
                        f"{state.neo4j_context_text[:2000]}\n\n"
                        f"Use the above ONLY if directly relevant to the user's current message. "
                        f"Always prioritize responding to what the user is saying NOW.\n\n"
                    )
                direct_prompt += (
                    f"The user said: \"{state.user_message}\"\n\n"
                    f"Respond naturally and helpfully, taking into account any retrieved user profile, preferences, or memory context if relevant. If they're greeting you, greet them back. "
                    f"If they ask what you can do, explain that you can connect them with specialized agents for:\n"
                    f"- Medical image analysis\n"
                    f"- Physics problems and explanations\n"
                    f"- Image generation\n"
                    f"- Multimodal analysis\n\n"
                    f"Keep your response brief and friendly."
                )
                
                direct_response = await ainvoke_with_retry(llm, [HumanMessage(content=direct_prompt)])
                state.agent_response = direct_response.content
                state.chosen_agent = "DIRECT"  # Mark that we responded directly
                print(f"✅ Direct response generated: {state.agent_response[:100]}...")
                return None  # End workflow
            
            # Validate the chosen agent exists
            agent_names = [agent['name'] for agent in state.available_agents]
            
            if chosen in agent_names:
                state.chosen_agent = chosen
                print(f"✅ Chose agent: {chosen}")
                return "send_to_agent"
            else:
                # Try to find a partial match
                chosen_lower = chosen.lower()
                for name in agent_names:
                    if name.lower() in chosen_lower or chosen_lower in name.lower():
                        state.chosen_agent = name
                        print(f"✅ Chose agent (partial match): {name} (from: {chosen})")
                        return "send_to_agent"
                
                # Default to first agent if no match
                state.chosen_agent = agent_names[0]
                print(f"⚠️ No match for '{chosen}', defaulting to: {state.chosen_agent}")
                return "send_to_agent"
                
        except Exception as e:
            print(f"❌ Error during classification: {str(e)}")
            import traceback
            traceback.print_exc()
            # Assign a sensible default instead of failing
            if state.available_agents:
                if state.has_images:
                    # Try to find the physics or medical agent for image queries
                    for agent in state.available_agents:
                        name_lower = agent['name'].lower()
                        if 'física' in name_lower or 'physics' in name_lower or 'multimodal' in name_lower:
                            state.chosen_agent = agent['name']
                            break
                    if not state.chosen_agent:
                        state.chosen_agent = state.available_agents[0]['name']
                else:
                    state.chosen_agent = state.available_agents[0]['name']
                print(f"🔄 Fallback: routing to {state.chosen_agent}")
                return "send_to_agent"
            state.error = f"Error during classification: {str(e)}"
            return None
    
    # Step 3: Send the message to the chosen agent
    @traceable(name="orchestrator_send_to_agent", run_type="chain", tags=["agent_type:orchestrator", "orchestrator"])
    async def send_to_agent(state: OrchestratorState) -> str:
        """Forward the user's message (with images if any) to the chosen agent"""
        
        # Check if we already have a direct response
        if state.agent_response:
            print(f"✅ Using direct response (no agent needed)")
            return None  # End workflow
        
        # Check if this was a direct response case
        if state.chosen_agent == "DIRECT":
            print(f"✅ Direct response already handled")
            return None  # End workflow
        
        print(f"📤 Step 3: Sending message to {state.chosen_agent}...")
        
        if not state.chosen_agent:
            # Try to recover by using the first available agent
            if state.available_agents:
                if state.has_images:
                    for agent in state.available_agents:
                        name_lower = agent['name'].lower()
                        if 'física' in name_lower or 'physics' in name_lower or 'multimodal' in name_lower:
                            state.chosen_agent = agent['name']
                            break
                if not state.chosen_agent and state.available_agents:
                    state.chosen_agent = state.available_agents[0]['name']
                print(f"🔄 Recovered: routing to {state.chosen_agent}")
            else:
                print(f"⚠️ No agent chosen and no agents available, generating fallback response")
                state.agent_response = "Lo siento, no pude determinar qué agente especializado usar para tu consulta. ¿Podrías reformular tu pregunta?"
                return None
        
        try:
            from service.server.beeai_host_manager import \
                SendMessageToAgentInput
            
            # Query NAMS context specifically for the chosen agent
            agent_context_text = ""
            if manager.neo4j_memory:
                await manager._ensure_neo4j_connected()
                if getattr(manager, '_neo4j_connected', False):
                    try:
                        student_id = state.student_id or state.context_id
                        print(f"🧠 Querying student NAMS context for student '{student_id}' and agent '{state.chosen_agent}'...")
                        ctx = await manager.get_student_context(
                            state.user_message,
                            student_id=student_id,
                            session_id=state.context_id,
                            agent_name=state.chosen_agent
                        )
                        if ctx:
                            raw_text = str(ctx)
                            ignore_headers = (
                                '## conversation history',
                                '### relevant past messages',
                                'conversation history',
                                'relevant past messages'
                            )
                            filtered_lines = []
                            in_chat_history_section = False
                            for line in raw_text.split('\n'):
                                line_stripped = line.strip()
                                line_lower = line_stripped.lower()
                                if not line_lower:
                                    continue
                                if any(h in line_lower for h in ignore_headers):
                                    in_chat_history_section = True
                                    continue
                                if '## relevant knowledge' in line_lower or '### user preferences' in line_lower:
                                    in_chat_history_section = False
                                    continue
                                if in_chat_history_section:
                                    continue
                                if any(line_lower.startswith(prefix) for prefix in [
                                    'user:', 'assistant:', 'human:', 'ai:',
                                    'usuario:', 'asistente:', 'q:', 'a:',
                                    '- [user]', '- [assistant]', '- [human]', '- [ai]',
                                    '- [usuario]', '- [asistente]'
                                ]):
                                    continue
                                filtered_lines.append(line)
                            agent_context_text = '\n'.join(filtered_lines).strip()
                            if len(agent_context_text) > 3000:
                                agent_context_text = agent_context_text[:3000] + "\n[... truncado]"
                    except Exception as e:
                        print(f"⚠️ Error retrieving Neo4j context for chosen agent {state.chosen_agent}: {e}")

            message_text = state.user_message
            if agent_context_text:
                message_text = f"[NAMS_CONTEXT]\n{agent_context_text}\n[/NAMS_CONTEXT]\n\n{state.user_message}"
                print(f"🧠 Injected agent-specific NAMS context into message sent to {state.chosen_agent}")
                
            send_input = SendMessageToAgentInput(
                agent_name=state.chosen_agent,
                message=message_text
            )
            
            result = await send_tool._run(send_input, None, None)
            state.agent_response = result
            print(f"✅ Agent responded")
            return None  # End workflow
            
        except Exception as e:
            state.error = f"Error communicating with agent: {str(e)}"
            print(f"❌ {state.error}")
            import traceback
            traceback.print_exc()
            return None
    
    # Create workflow with the state schema and name
    workflow = Workflow(schema=OrchestratorState, name="AgentOrchestrator")
    
    # Add steps using add_step() method
    workflow.add_step("list_agents", list_agents)
    workflow.add_step("classify_and_choose", classify_and_choose)
    workflow.add_step("send_to_agent", send_to_agent)
    
    # Set the starting step
    workflow.set_start("list_agents")
    
    return workflow
