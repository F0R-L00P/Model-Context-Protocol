# FunctionsAndTools.py
import os
import re
import time
import logging

from duckduckgo_search import DDGS
from typing_extensions import TypedDict
from typing import List, Optional, TypedDict, Any

from llm_provider import LLMClient
from langchain.globals import set_debug
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

#set_debug(True)
############################################
# Configure logger
############################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s • %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

############################################
# initialize a “generic” LLM client 
############################################

# 1) State definition
class Message(TypedDict):
    role: str
    content: str

# 2) LLM
llm = LLMClient()

# 3) agent
class AgentState(TypedDict):
    history: List[Message]          # full chat history
    plan: Optional[str]             # planner's current plan
    search_queries: List[str]       # queries to run
    search_results: List[str]       # raw snippets/urls
    final_answer: Optional[str]     # what we return to the user

# 4) Graph
builder = StateGraph(State) 
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()

user_message = {
    "role": "user",
    "content": "Hey Buddy, whats new?"
}

initial_state = {
    "messages":[user_message]
}

graph.invoke(initial_state)