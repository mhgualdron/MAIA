from dotenv import load_dotenv
import os

load_dotenv()

try:
    from langchain_tavily import TavilySearch
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_openai import ChatOpenAI
except ImportError:
    raise ImportError(
        "Faltan dependencias de LangChain/LangGraph. "
    )

memory = MemorySaver()
tavily_tool = TavilySearch(max_results=2)
tools = [tavily_tool]

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
llm_with_tools = llm.bind_tools(tools)

graph_config = {"configurable":{"thread_id":"1"}}



