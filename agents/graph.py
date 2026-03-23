"""
LangGraph graph definition.

Graph structure:
  START → hyde_node → retrieve_node → answer_node → END

HyDE is always applied — experiments confirmed that always generating a
hypothetical document before retrieval outperforms conditional routing
(see notebooks/rag_experiments.ipynb, Experiment 2).
"""

from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.nodes import AgentNodes


def build_graph(nodes: AgentNodes):
    """
    Build and compile the multi-agent LangGraph.

    Args:
        nodes: Initialised AgentNodes instance carrying all dependencies.
    """
    builder = StateGraph(AgentState)

    builder.add_node("hyde_node", nodes.hyde_node)
    builder.add_node("retrieve_node", nodes.retrieve_node)
    builder.add_node("answer_node", nodes.answer_node)

    builder.add_edge(START, "hyde_node")
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
