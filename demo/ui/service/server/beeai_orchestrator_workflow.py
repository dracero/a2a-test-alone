"""
BeeAI Workflow-based Orchestrator
Compatible with Groq (Llama 4)
Uses explicit workflow steps instead of ReAct pattern
"""

import json
from typing import Any

from beeai_framework.workflows.workflow import Workflow
from pydantic import BaseModel


class OrchestratorState(BaseModel):
    """State for the orchestration workflow"""
    user_message: str
    has_images: bool
    available_agents: list[dict] = []
    chosen_agent: str = ""
    agent_response: str = ""
    error: str = ""


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
    async def classify_and_choose(state: OrchestratorState) -> str:
        """Use Groq to analyze the request and choose the best agent or respond directly"""
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
            
            # Create a classification prompt for Groq
            classification_prompt = (
                f"You are a routing system. Analyze the user's request and decide if it needs a specialized agent or if you should respond directly.\n\n"
                f"Available specialized agents:\n{agents_description}\n\n"
                f"User request: \"{state.user_message}\"\n"
                f"Request includes images: {state.has_images}\n\n"
                f"RULES:\n"
                f"1. If the request is a simple greeting (hello, hi, hola, hey, etc.), respond with: DIRECT\n"
                f"2. If the request is small talk or general conversation, respond with: DIRECT\n"
                f"3. If the request asks what you can do or who you are, respond with: DIRECT\n"
                f"4. If the request needs specialized knowledge or analysis, respond with the EXACT agent name.\n\n"
                f"Examples:\n"
                f"- 'Hola' → DIRECT\n"
                f"- 'Hello' → DIRECT\n"
                f"- 'Hi there' → DIRECT\n"
                f"- 'How are you?' → DIRECT\n"
                f"- 'What can you do?' → DIRECT\n"
                f"- 'Analyze this medical image' → Asistente Médico\n"
                f"- 'Help me with physics homework' → Agente de Física - Problemas e Imágenes\n"
                f"- 'Generate an image of a cat' → Image Generator Agent\n\n"
                f"IMPORTANT: Respond with ONLY ONE WORD - either 'DIRECT' or the exact agent name. Nothing else."
            )
            
            # Use Groq to classify
            from langchain_core.messages import HumanMessage
            
            print(f"🔍 Sending classification prompt to Groq...")
            llm_response = await llm.ainvoke([HumanMessage(content=classification_prompt)])
            
            print(f"📥 Groq response type: {type(llm_response)}")
            print(f"📥 Groq response content: {llm_response.content}")
            
            chosen = str(llm_response.content).strip() if llm_response.content else ""
            
            if not chosen:
                print(f"⚠️ Groq returned empty response, using first agent")
                state.chosen_agent = state.available_agents[0]['name']
                return "send_to_agent"
            
            print(f"🎯 Raw chosen: '{chosen}'")
            
            # Clean up the response
            chosen = chosen.replace('"', '').replace("'", '').strip()
            if chosen and chosen[0].isdigit():
                parts = chosen.split('.', 1)
                if len(parts) > 1:
                    chosen = parts[1].strip()
            
            print(f"🎯 Cleaned chosen: '{chosen}'")
            
            # Check if should respond directly
            if chosen.upper() == 'DIRECT':
                print(f"✅ Responding directly (no agent needed)")
                # Generate a direct response
                direct_prompt = (
                    f"You are a friendly AI assistant. The user said: \"{state.user_message}\"\n\n"
                    f"Respond naturally and helpfully. If they're greeting you, greet them back. "
                    f"If they ask what you can do, explain that you can connect them with specialized agents for:\n"
                    f"- Medical image analysis\n"
                    f"- Physics problems and explanations\n"
                    f"- Image generation\n"
                    f"- Multimodal analysis\n\n"
                    f"Keep your response brief and friendly."
                )
                
                direct_response = await llm.ainvoke([HumanMessage(content=direct_prompt)])
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
            state.error = f"Error during classification: {str(e)}"
            print(f"❌ {state.error}")
            return None
    
    # Step 3: Send the message to the chosen agent
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
            # This should not happen if classify_and_choose worked correctly
            # Generate a fallback response
            print(f"⚠️ No agent chosen, generating fallback response")
            state.agent_response = "Lo siento, no pude determinar qué agente especializado usar para tu consulta. ¿Podrías reformular tu pregunta?"
            return None
        
        try:
            from service.server.beeai_host_manager import \
                SendMessageToAgentInput
            send_input = SendMessageToAgentInput(
                agent_name=state.chosen_agent,
                message=state.user_message
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
