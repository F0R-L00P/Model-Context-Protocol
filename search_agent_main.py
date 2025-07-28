# search_agent_main.py
"""
Adapted from https://github.com/Sujar-Henry/AI_Search_Agent/blob/main/search_agent/main.py
"""
import logging
import os
from typing import List
from llm_provider import LLMClient

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s • %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize LLM client
llm = LLMClient()
logger.info("✅ LLMClient wrapper initialized.")

def search_agent(query: str, history: List[str] = None) -> str:
    """
    Simple search agent using Watsonx LLM.
    Args:
        query (str): The user's search query.
        history (List[str], optional): Previous conversation history.
    Returns:
        str: LLM response.
    """
    messages = []
    if history:
        for msg in history:
            messages.append({"role": "user", "content": msg})
    messages.append({"role": "user", "content": query})
    response = llm.invoke(messages)
    return response["content"] if isinstance(response, dict) and "content" in response else str(response)

if __name__ == "__main__":
    # Example usage
    user_query = "What is the capital of France?"
    result = search_agent(user_query)
    print("Agent response:", result)
