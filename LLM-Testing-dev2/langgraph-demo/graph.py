# 1. Importación Correcta de State (ya definida en state.py)
from state import State

# 2. Importaciones de Config y LangGraph
from config import tools, memory, llm_with_tools, llm 
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.graph import END, StateGraph

# --- Función de Construcción del Grafo ---
def build_graph() -> StateGraph:
    graph = StateGraph(State)
    def chatbot(state: State) -> State:
        message = llm_with_tools.invoke(state["messages"])
        return {"messages": [message]}
    
    graph.add_node('Chatbot', chatbot)
    tool_node = ToolNode(tools)
    graph.add_node('ToolNode', tool_node)

    graph.add_conditional_edges(
    'Chatbot',
    tools_condition,
    {
        # Si tools_condition retorna "tools" (quiere llamar herramienta), va a 'ToolNode'
        "tools": "ToolNode", 
        # Si tools_condition retorna "END" (respuesta final), el grafo termina
        "__end__": END
    }
)
    
    graph.add_edge('ToolNode', 'Chatbot')


    graph.set_entry_point('Chatbot') 
    
    # Compilar
    return graph.compile(checkpointer=memory)