"""All prompt templates for the multi-agent RAG system."""

from langchain_core.prompts import ChatPromptTemplate

# --- Router Agent ---
# Decides whether to use HyDE query rewriting. LLM makes the decision —
# no hard-coded keyword matching.
ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """You are an intelligent query router for a social media search system.

Your job is to decide whether to rewrite the user's question into a hypothetical
social media post (HyDE) before searching, or to search directly.

Use HyDE (set use_hyde to true) when:
- The question asks about opinions, reactions, sentiment, or feelings
- The question is vague or conversational in style
- The question would benefit from being rephrased as a social media post

Use direct retrieval (set use_hyde to false) when:
- The question mentions specific named entities (people, places, events, companies)
- The question asks about specific facts or news items
- The question is already specific and keyword-rich

Question: {question}

Respond with ONLY a JSON object in this exact format:
{{"use_hyde": true, "reason": "brief explanation"}}
or
{{"use_hyde": false, "reason": "brief explanation"}}"""
)

# --- HyDE Agent ---
# Generates a hypothetical social media post that would answer the question.
HYDE_PROMPT = ChatPromptTemplate.from_template(
    """You are a social media user writing a post in December 2024.
Write a realistic social media post that would answer the following question.
Keep it short (1-3 sentences), casual, and in the style of a real social media post.

Question: {question}

Hypothetical social media post:"""
)
