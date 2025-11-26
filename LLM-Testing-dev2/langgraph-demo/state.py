from typing_extensions import TypedDict
from typing import List, Annotated
from langgraph.graph.message import AnyMessage, add_messages

class State(TypedDict):
    """
    Representa el estado del gráfico.
    
    messages: Historial de mensajes de la conversación.
    """
    messages: Annotated[List[AnyMessage], add_messages]
