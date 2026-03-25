"""
Builds and compiles the LangGraph pipeline.

The graph is a straight sequence: START -> hyde -> retrieve -> answer -> END.
Conditional routing was tested in Experiment 2 and didn't help, so this
is intentionally a fixed pipeline rather than a dynamic agent.
"""

import logging

from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.nodes import AgentNodes

logger = logging.getLogger(__name__)


def build_graph(nodes: AgentNodes):
    """Build and compile the graph. Takes an AgentNodes instance with all dependencies loaded."""
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
    """Render the graph as a PNG and save it. Used by the Streamlit sidebar."""
    try:
        from langchain_core.runnables.graph import MermaidDrawMethod
        img_bytes = graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API
        )
        with open(path, "wb") as f:
            f.write(img_bytes)
        logger.info("Graph saved to %s", path)
    except Exception as e:
        logger.warning("Could not save graph image: %s", e)
