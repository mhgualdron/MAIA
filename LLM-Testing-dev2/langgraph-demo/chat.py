from graph import build_graph
from config import graph_config
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AnyMessage

def stream_graph_updates(user_input: str):
    graph = build_graph()

    for event in graph.stream(
        {'messages': [{'role':'user', 'content': user_input}]},
        config=graph_config,
        stream_mode = 'values'
    ):
        print(event['messages'][-1])

def run_chat_loop():
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting chat. Goodbye!")
                break
            stream_graph_updates(user_input)
        except KeyboardInterrupt:
            print("\nExiting chat. Exception Raised.")
            break