import ast
import json
import operator
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL = "openai/gpt-oss-120b"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment.")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in the environment.")

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_math(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_math(node.left)
        right = _evaluate_math(node.right)
        return _BINARY_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_math(node.operand))

    raise ValueError("Only numeric arithmetic expressions are supported.")


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression without executing arbitrary code."""
    try:
        parsed_expression = ast.parse(expression.strip(), mode="eval")
        result = _evaluate_math(parsed_expression.body)
        return str(result)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
        return f"Calculation error: {error}"


def web_search(query: str) -> str:
    """Search the web with Tavily and return its answer with supporting sources."""
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            include_answer=True,
            max_results=5,
        )
        answer = response.get("answer")
        results = response.get("results", [])

        if answer:
            return answer

        if not results:
            return "No relevant web results were found."

        return "\n".join(
            f"- {result.get('title', 'Untitled')}: {result.get('content', '')}"
            for result in results
        )
    except Exception as error:
        return f"Web search error: {error}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for current or factual information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or search query to look up.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate a basic arithmetic expression such as 2+2.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A numeric arithmetic expression.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

_tool_functions = {
    "web_search": web_search,
    "calculate": calculate,
}


def run_agent(question: str) -> str:
    """Let Groq choose tools automatically and return the final answer."""
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use web_search for current or "
                "external information and calculate for arithmetic. Use tools "
                "when needed, then answer the user's question clearly."
            ),
        },
        {"role": "user", "content": question},
    ]

    for _ in range(10):
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        messages.append(assistant_message.model_dump(exclude_none=True))

        if not tool_calls:
            return assistant_message.content or ""

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function = _tool_functions.get(function_name)
            if function is None:
                tool_result = f"Unknown tool: {function_name}"
            else:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    tool_result = function(**arguments)
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    tool_result = f"Tool input error: {error}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result),
                }
            )

    raise RuntimeError("The agent exceeded the maximum number of tool-call rounds.")


if __name__ == "__main__":
    user_question = input("Ask a question: ").strip()
    if not user_question:
        raise ValueError("Please enter a question.")
    print(run_agent(user_question))
