from state import State

from config import *
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.graph import END, StateGraph
# --- Graph Construction Function ---


def build_graph() -> StateGraph:
    graph = StateGraph(State)

    def chatbot(state: State) -> State:
        message = llm_with_tools.invoke(state["messages"])
        return {"messages": [message]}

    #Node Definitions
    
    graph.add_node('Chatbot', chatbot)
    tool_node = ToolNode(tools)
    graph.add_node('ToolNode', tool_node)

    graph.add_conditional_edges(
        'Chatbot',
        tools_condition,
        {
            # If tools_condition returns "tools" (wants to call tool), go to 'ToolNode'
            "tools": "ToolNode",
            # If tools_condition returns "END" (final response), end the graph
            "__end__": END
        }
    )

    graph.add_edge('ToolNode', 'Chatbot')

    graph.set_entry_point('Chatbot')

    # Compile
    return graph.compile(checkpointer=memory)