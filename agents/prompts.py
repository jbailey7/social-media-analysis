"""All prompt templates for the multi-agent RAG system."""

from langchain_core.prompts import ChatPromptTemplate

# --- HyDE Agent ---
# Generates a hypothetical social media post matching the query so that
# embedding similarity targets post-like text rather than question-like text.
HYDE_PROMPT = ChatPromptTemplate.from_template(
    """You are a social media user writing a post in December 2024.
Write a realistic social media post that would answer the following question.
Keep it short (1-3 sentences), casual, and in the style of a real social media post.

Question: {question}

Hypothetical social media post:"""
)
