"""Prompt templates."""

from langchain_core.prompts import ChatPromptTemplate

# HyDE: turn the user's question into a fake social media post.
# This works better than embedding the raw question because the index
# contains posts, not questions — so the similarity search is more useful.
HYDE_PROMPT = ChatPromptTemplate.from_template(
    """You are a social media user writing a post in December 2024.
Write a realistic social media post that would answer the following question.
Keep it short (1-3 sentences), casual, and in the style of a real social media post.

Question: {question}

Hypothetical social media post:"""
)
