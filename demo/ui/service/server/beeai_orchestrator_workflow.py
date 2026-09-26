"""
BeeAI Workflow-based Orchestrator
Powered by JEV (TypeSafe AI System One)
Uses explicit workflow steps instead of ReAct pattern
"""

import json
from typing import Any

from beeai_framework.workflows.workflow import Workflow
from pydantic import BaseModel

from .api_key_rotator import ainvoke_with_retry, sanitize_nams_context
from .jev_service import classify_agent_routing
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
    
    # Step 2: Use JEV (TypeSafe AI System One) to classify and choose the best agent
    @traceable(name="orchestrator_classify_and_choose", run_type="chain", tags=["agent_type:orchestrator", "orchestrator"])
    async def classify_and_choose(state: OrchestratorState) -> str:
        """Use JEV System One model to evaluate the request and choose the best agent or DIRECT."""
        print("🤔 Step 2: Classifying request and choosing agent with JEV (TypeSafe AI)...")
        
        if not state.available_agents:
            state.error = "No agents available"
            return None
        
        try:
            chosen, confidence, probs = await classify_agent_routing(
                user_message=state.user_message,
                available_agents=state.available_agents,
                history_text=state.history_text,
                neo4j_context_text=state.neo4j_context_text,
                has_images=state.has_images,
                image_count=len(state.image_data_list) if state.has_images else 0,
            )
            print(f"🎯 [JEV] Chosen route: '{chosen}' (confidence: {confidence:.2f})")
            
            # Check if should respond directly (greetings, small talk, general info)
            if chosen.upper() == 'DIRECT':
                print(f"✅ Responding directly (no specialized agent needed)")
                # Generate a direct conversational response
                from langchain_core.messages import HumanMessage
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
                    f"Respond naturally and helpfully in Spanish, taking into account any retrieved user profile, preferences, or memory context if relevant. If they're greeting you, greet them back. "
                    f"If they ask what you can do, explain that you can connect them with specialized agents for:\n"
                    f"- Análisis de imágenes histológicas y medicina (Asistente Médico)\n"
                    f"- Problemas y explicaciones de física con método socrático (Tutor Socrático de Física Multimodal)\n"
                    f"- Generación de imágenes artísticas (Image Generator Agent)\n\n"
                    f"Keep your response brief and friendly."
                )
                
                if llm is not None:
                    try:
                        direct_response = await ainvoke_with_retry(llm, [HumanMessage(content=direct_prompt)])
                        state.agent_response = direct_response.content
                    except Exception as e:
                        print(f"⚠️ Error generating direct response via LLM: {e}")
                        state.agent_response = (
                            "¡Hola! Soy tu asistente de aprendizaje. Puedo ayudarte conectándote con nuestros agentes especializados:\n"
                            "- **Asistente Médico**: Consultas de histopatología y búsqueda de micrografías.\n"
                            "- **Tutor Socrático de Física Multimodal**: Resolución paso a paso de problemas de física.\n"
                            "- **Image Generator Agent**: Creación de imágenes digitales y esquemas.\n\n"
                            "¿En qué te gustaría profundizar hoy?"
                        )
                else:
                    state.agent_response = (
                        "¡Hola! Soy tu asistente de aprendizaje. Puedo ayudarte conectándote con nuestros agentes especializados:\n"
                        "- **Asistente Médico**: Consultas de histopatología y búsqueda de micrografías.\n"
                        "- **Tutor Socrático de Física Multimodal**: Resolución paso a paso de problemas de física.\n"
                        "- **Image Generator Agent**: Creación de imágenes digitales y esquemas.\n\n"
                        "¿En qué te gustaría profundizar hoy?"
                    )
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
                print(f"⚠️ No exact match for '{chosen}', defaulting to: {state.chosen_agent}")
                return "send_to_agent"
                
        except Exception as e:
            print(f"❌ Error during JEV classification: {str(e)}")
            import traceback
            traceback.print_exc()
            # Assign a sensible default instead of failing
            if state.available_agents:
                if state.has_images:
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
                    agent_context_text = sanitize_nams_context(raw_text)
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
