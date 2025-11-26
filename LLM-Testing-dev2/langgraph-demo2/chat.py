# chat.py
from graph import build_graph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AnyMessage
from config import graph_config, memory, llm # Importa las configuraciones necesarias

# Inicializa el grafo una sola vez
AGENT_GRAPH = build_graph()

def process_user_message(user_input: str, thread_id: str = "1") -> str:
    """
    Procesa un solo mensaje del usuario usando el grafo con checkpointer.
    """
    
    # 1. Configurar el thread_id para el checkpointer (memoria)
    config = {"configurable": {"thread_id": thread_id}}

    # 2. Definir el input del grafo
    # El estado inicial es el mensaje del usuario
    agent_input = {"messages": [HumanMessage(content=user_input)]}

    # 3. Invocar al grafo
    final_state = AGENT_GRAPH.invoke(
        agent_input,
        config=config,
    )
    
    # 4. Extraer el último mensaje (la respuesta del AI)
    last_message = final_state.get("messages", [])[-1]
    
    if isinstance(last_message, (AIMessage, SystemMessage)):
        return last_message.content
    elif isinstance(last_message, HumanMessage):

        return f"Error: El agente respondió con un mensaje humano. Contenido: {last_message.content}"
    else:
         return str(last_message)
