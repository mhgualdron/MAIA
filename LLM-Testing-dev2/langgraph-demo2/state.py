from typing_extensions import TypedDict
from typing import List, Annotated
from langgraph.graph.message import AnyMessage, add_messages

class State(TypedDict):
    """
    Represents the state of the graph.
    
    messages: Conversation message history.
    """
    messages: Annotated[List[AnyMessage], add_messages]