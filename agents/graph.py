"""
LangGraph graph definition.

Graph structure:
  START → router_node
  router_node --[use_hyde=True]--> hyde_node --> retrieve_node --> answer_node --> END
  router_node --[use_hyde=False]-------------> retrieve_node --> answer_node --> END
"""

from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.nodes import router_node, hyde_node, retrieve_node, answer_node, route_after_router


def build_graph():
    """Build and compile the multi-agent LangGraph."""
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("router_node", router_node)
    builder.add_node("hyde_node", hyde_node)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("answer_node", answer_node)

    # Edges
    builder.add_edge(START, "router_node")
    builder.add_conditional_edges(
        "router_node",
        route_after_router,
        {"hyde_node": "hyde_node", "retrieve_node": "retrieve_node"},
    )
    builder.add_edge("hyde_node", "retrieve_node")
    builder.add_edge("retrieve_node", "answer_node")
    builder.add_edge("answer_node", END)

    return builder.compile()


def save_graph_image(graph, path: str = "graph.png"):
    """Save a Mermaid-rendered PNG of the compiled graph."""
    try:
        from langchain_core.runnables.graph import MermaidDrawMethod
        img_bytes = graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API
        )
        with open(path, "wb") as f:
            f.write(img_bytes)
        print(f"Graph saved to {path}")
    except Exception as e:
        print(f"Could not save graph image: {e}")


# Compile at import time so app.py can import `compiled_graph` directly
compiled_graph = build_graph()
