# FunctionsAndTools.py

import logging
from duckduckgo_search import DDGS
from typing_extensions import TypedDict
from typing import List, Optional
from llm_provider import LLMClient
from langgraph.graph import StateGraph, END

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s • %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# State definition
class Message(TypedDict):
    role: str
    content: str

class AgentState(TypedDict):
    history: List[Message]
    plan: Optional[str]
    search_queries: List[str]
    search_results: List[str]
    final_answer: Optional[str]

# LLM
llm = LLMClient()

# DuckDuckGo search tool with deduplication
def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """
    Returns a joined string of result titles + snippets + urls, deduplicated by URL.
    """
    collected = []
    seen_urls = set()
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        for item in results:
            url = item.get("href", "")
            if url and url in seen_urls:
                continue
            seen_urls.add(url)
            title = item.get("title", "")
            snippet = item.get("body", "")
            collected.append(f"{title}\n{snippet}\n{url}")
    return "\n\n".join(collected)


#####################################################################
## models
####################################################################
def planner_node(state: AgentState) -> AgentState:
    """
    Decide: Do we need to search? Produce a plan + (optional) queries.
    """
    import json
    system = (
        "You are a planner. Decide if web search is needed. "
        "If needed, produce 1-3 concise DuckDuckGo queries. "
        "Return JSON with keys: plan, need_search (true/false), queries (list)."
    )
    messages = [{"role": "system", "content": system}] + state["history"]
    response = llm.invoke(messages).content
    logger.info("Planner RAW:\n%s", response)
    try:
        parsed = json.loads(response)
        state["plan"] = parsed.get("plan", "")
        state["search_queries"] = parsed.get("queries", [])
    except Exception:
        state["plan"] = response
        state["search_queries"] = []
    return state


def search_node(state: AgentState) -> AgentState:
    """
    Run DuckDuckGo for each query, store raw results.
    """
    raw_results = []
    for q in state["search_queries"]:
        block = duckduckgo_search(q, max_results=5)
        if block:
            raw_results.append(f"### Results for: {q}\n{block}")
    state["search_results"] = raw_results
    return state


def summarize_node(state: AgentState) -> AgentState:
    """
    Summarize the search results + original question into a final answer.
    """
    prompt = (
        "You are a helpful assistant. Use the SEARCH RESULTS below to answer the USER.\n"
        "Cite sources inline as [#] where # is an index of the block.\n"
        "If something isn't found, say so.\n\n"
        "USER QUESTION:\n"
    )
    user_msg = next((m["content"] for m in reversed(state["history"]) if m["role"] == "user"), "")
    prompt += user_msg + "\n\nSEARCH RESULTS:\n"
    for i, block in enumerate(state["search_results"]):
        prompt += f"[{i}] {block}\n\n"
    result = llm.invoke([{"role": "user", "content": prompt}]).content
    state["final_answer"] = result
    return state


def finalize_node(state: AgentState) -> AgentState:
    """
    Post-step guard. If we already have an answer, keep it. Otherwise, answer directly (no-search path).
    """
    if (state.get("final_answer") or "").strip():
        return state
    question = next((m.get("content", "") for m in reversed(state.get("history", [])) if m.get("role") == "user"), "")
    answer = llm.invoke([
        {"role": "system", "content": "Answer the user clearly and concisely."},
        {"role": "user", "content": question},
    ]).content
    state["final_answer"] = answer
    return state


#########################################
# 5) build graph
############################################
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("finalize", finalize_node)
    graph.set_entry_point("planner")
    def needs_search(state: AgentState) -> str:
        return "search" if state.get("search_queries") else "finalize"
    graph.add_conditional_edges("planner", needs_search, {"search": "search", "finalize": "finalize"})
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


###################################################
# 6) call function
##################################################
def reset_state(state: AgentState):
    state["plan"] = None
    state["search_queries"] = []
    state["search_results"] = []
    state["final_answer"] = None

def main():
    app = build_graph()
    state: AgentState = {
        "history": [],
        "plan": None,
        "search_queries": [],
        "search_results": [],
        "final_answer": None,
    }
    print("Ask me something (Ctrl+C to quit):")
    while True:
        try:
            user_input = input("> ")
        except KeyboardInterrupt:
            print("\nBye.")
            break
        msg: Message = {"role": "user", "content": user_input}
        state["history"].append(msg)
        state = app.invoke(state)
        print("\n--- Answer ---")
        print(state["final_answer"])
        print("--------------\n")
        reset_state(state)

if __name__ == "__main__":
    main()