"""
LangGraph travel agent.

Graph:
    START -> call_model (tool calling) or plan_fallback (models without tool support)
    call_model -> tools -> call_model ... -> respond
    plan_fallback -> tools -> respond
    respond -> END

Node names avoid the word "agent": OpenInference marks any run whose name
contains "agent" as an AGENT span, and only the graph root (travel_agent)
should be one.

The call_model node lets the model choose tools. When the model cannot call tools,
or stops before calling all of them, plan_fallback issues the missing tool calls
through the same ToolNode, so every path produces the same TOOL spans. The
respond node always writes the final answer from the collected tool results, and
returns a fixed answer when the knowledge base has no information.

Paths recorded in state["path"]: tool_calling (model called every tool), hybrid
(model called some tools, fallback ran the rest), fallback (model called none).
"""

import logging
import uuid
from pathlib import Path
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agent.tools import match_known_destination

logger = logging.getLogger(__name__)

AGENT_NAME = "travel_agent"

DEFAULT_SYSTEM_PROMPT = """You are a travel advisor agent with tools. For the destination in the user request, call validate_destination, search_destination_kb and get_current_season. Pass only the destination name as the argument. Do not answer from memory."""

NO_INFORMATION_ANSWER = "I don't have information about that destination"

DEFAULT_RESPOND_PROMPT = """You are a helpful travel advisor assistant. Use ONLY the tool results below to answer the user request.

CRITICAL INSTRUCTIONS:
- Use ONLY the facts from the tool results provided below
- Tailor the advice to the current season when the season is known
- Keep your response concise and helpful, maximum 50 words

<tool_results>
{tool_results}
</tool_results>

User request: {request}

Travel Advice:"""


class AgentState(TypedDict):
    """State passed between graph nodes."""
    messages: Annotated[List[AnyMessage], add_messages]
    request: str
    path: str
    steps: int
    answer: str


def load_agentic_prompts(prompt_path: str) -> tuple[str, str]:
    """Load the SYSTEM and RESPOND prompts from the filesystem, with code fallbacks."""
    path = Path(prompt_path)
    if not path.exists():
        logger.warning(f"Agentic prompt file not found at {path}; using built-in fallback prompts")
        return DEFAULT_SYSTEM_PROMPT, DEFAULT_RESPOND_PROMPT

    sections = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            current = line[4:].strip().upper()
            sections[current] = []
        elif current:
            sections[current].append(line)

    system_prompt = "\n".join(sections.get("SYSTEM", [])).strip() or DEFAULT_SYSTEM_PROMPT
    respond_prompt = "\n".join(sections.get("RESPOND", [])).strip() or DEFAULT_RESPOND_PROMPT
    return system_prompt, respond_prompt


def model_supports_tools(ollama_client, model: str) -> bool:
    """Return True when the Ollama model advertises the 'tools' capability."""
    try:
        info = ollama_client.show(model)
        capabilities = info.get("capabilities") if hasattr(info, "get") else getattr(info, "capabilities", None)
        supported = "tools" in (capabilities or [])
        logger.info(f"Model '{model}' tool calling support: {supported}")
        return supported
    except Exception as e:
        logger.warning(f"Could not determine tool support for model '{model}': {e}")
        return False


def _called_tools(messages: List[AnyMessage]) -> set:
    """Names of the tools that have already returned results."""
    return {msg.name for msg in messages if isinstance(msg, ToolMessage)}


def _tool_result(messages: List[AnyMessage], name: str) -> str:
    """Latest result of a tool, or an empty string if it has not run."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == name:
            return str(msg.content)
    return ""


def _model_destination(messages: List[AnyMessage]) -> str:
    """Destination argument from the first tool call the model made, if any."""
    for msg in messages:
        for tool_call in getattr(msg, "tool_calls", None) or []:
            destination = (tool_call.get("args") or {}).get("destination")
            if destination:
                return str(destination)
    return ""


def _format_tool_results(messages: List[AnyMessage]) -> str:
    """Collect tool outputs from the conversation for the respond prompt."""
    results = [
        f"[{msg.name}]\n{msg.content}"
        for msg in messages
        if isinstance(msg, ToolMessage)
    ]
    return "\n\n".join(results) if results else "No tool results."


def build_agent_graph(
    chat_model,
    tools: List[BaseTool],
    known_destinations: List[str],
    system_prompt: str,
    respond_prompt: str,
    supports_tools: bool,
    max_steps: int,
):
    """Build and compile the travel agent graph."""
    tool_names = [t.name for t in tools]
    # Tool selection runs at temperature 0 for more reliable tool calls on small models;
    # the respond node uses the configured AI_TEMPERATURE.
    agent_model = (
        chat_model.model_copy(update={"temperature": 0}).bind_tools(tools) if supports_tools else None
    )

    def route_start(state: AgentState) -> str:
        return "call_model" if supports_tools else "plan_fallback"

    def call_model(state: AgentState) -> dict:
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        response = agent_model.invoke(messages)
        return {"messages": [response], "path": "tool_calling", "steps": state.get("steps", 0) + 1}

    def route_model(state: AgentState) -> str:
        last = state["messages"][-1]
        called = _called_tools(state["messages"])
        requested = {tc["name"] for tc in getattr(last, "tool_calls", None) or []}
        # Continue the loop only while the model asks for tools it has not used yet.
        if requested - called and state.get("steps", 0) <= max_steps:
            return "tools"
        if set(tool_names) - called:
            logger.info(f"Agent stopped before calling {sorted(set(tool_names) - called)}; running them as fallback")
            return "plan_fallback"
        return "respond"

    def plan_fallback(state: AgentState) -> dict:
        called = _called_tools(state["messages"])
        destination = (
            _model_destination(state["messages"])
            or match_known_destination(state["request"], known_destinations)
            or state["request"]
        )
        tool_calls = [
            {"name": name, "args": {"destination": destination}, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}
            for name in tool_names
            if name not in called
        ]
        # "hybrid": the model called some tools itself; "fallback": it called none.
        path = "hybrid" if called else "fallback"
        return {"messages": [AIMessage(content="", tool_calls=tool_calls)], "path": path}

    def route_tools(state: AgentState) -> str:
        return "respond" if state.get("path") in ("fallback", "hybrid") else "call_model"

    def respond(state: AgentState) -> dict:
        # Without knowledge base content, answer deterministically instead of letting
        # a small model improvise (same behavior as the RAG workflow).
        if _tool_result(state["messages"], "search_destination_kb").startswith("NO INFORMATION"):
            logger.info("Knowledge base has no information for the request; skipping LLM answer")
            return {"answer": NO_INFORMATION_ANSWER}

        prompt = respond_prompt.format(
            tool_results=_format_tool_results(state["messages"]),
            request=state["request"],
        )
        response = chat_model.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        return {"answer": answer}

    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("plan_fallback", plan_fallback)
    graph.add_node("respond", respond)

    graph.add_conditional_edges(START, route_start, ["call_model", "plan_fallback"])
    graph.add_conditional_edges("call_model", route_model, ["tools", "plan_fallback", "respond"])
    graph.add_edge("plan_fallback", "tools")
    graph.add_conditional_edges("tools", route_tools, ["call_model", "respond"])
    graph.add_edge("respond", END)

    return graph.compile(name=AGENT_NAME)


def run_agent(graph, request: str, max_steps: int) -> dict:
    """Run the travel agent for a user request and return the final state."""
    initial_state = {
        "messages": [HumanMessage(content=request)],
        "request": request,
        "path": "",
        "steps": 0,
        "answer": "",
    }
    # Run metadata is inherited by every child run (graph nodes, model calls, tools), so
    # OpenInference records agent_name in the `metadata` attribute of each span in the run.
    return graph.invoke(
        initial_state,
        config={
            "recursion_limit": max_steps * 2 + 6,
            "run_name": AGENT_NAME,
            "metadata": {"agent_name": AGENT_NAME},
        },
    )
