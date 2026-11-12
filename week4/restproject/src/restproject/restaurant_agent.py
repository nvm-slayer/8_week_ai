from __future__ import annotations

import os
import random
import re
from typing import Annotated, Callable, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

MENU = {
    "burger": 10,
    "pizza": 20,
    "momos": 4,
    "malai_chaap": 50,
}

Status = Literal[
    "new",
    "invalid",
    "confirmed",
    "partial",
    "unavailable",
    "ready",
    "complete",
    "failed",
    "awaiting_user",
]


class OrderRequest(BaseModel):
    dish: str | None = Field(description="The requested menu item, or null if unrelated.")
    quantity: int | None = Field(
        description="A positive requested quantity, or null if no food order was given."
    )
    action: Literal["new_order", "accept_partial", "decline", "invalid"] = Field(
        description="What the user wants to do with the current order."
    )


class OrderState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str
    dish: str
    required_quantity: int
    available_quantity: str
    status: Status
    order_attempts: int
    cook_attempts: int
    serve_attempts: int
    final_result: str
    response: str


def _normalise_dish(dish: str | None) -> str | None:
    if not dish:
        return None
    normalised = re.sub(r"[\s-]+", "_", dish.strip().lower())
    aliases = {"chaap": "malai_chaap", "malai chaap": "malai_chaap"}
    normalised = aliases.get(normalised, normalised)
    return normalised if normalised in MENU else normalised


def _fallback_extract(text: str) -> OrderRequest:
    lowered = text.lower()
    if any(word in lowered for word in ("no", "decline", "cancel", "stop")):
        return OrderRequest(dish=None, quantity=None, action="decline")
    if any(word in lowered for word in ("yes", "accept", "go ahead", "okay", "ok")):
        return OrderRequest(dish=None, quantity=None, action="accept_partial")

    quantity_match = re.search(r"\b(\d+)\b", lowered)
    dish = next((name for name in MENU if name.replace("_", " ") in lowered), None)
    if dish is None:
        return OrderRequest(dish=None, quantity=None, action="invalid")
    quantity = int(quantity_match.group(1)) if quantity_match else None
    if quantity is None or quantity <= 0:
        return OrderRequest(dish=None, quantity=None, action="invalid")
    return OrderRequest(dish=dish, quantity=quantity, action="new_order")


def extract_order(text: str) -> OrderRequest:
    """Extract an order with Groq, using a small deterministic parser offline."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fallback_extract(text)

    model = ChatGroq(
        api_key=api_key,
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
    )
    structured_model = model.with_structured_output(OrderRequest)
    result = structured_model.invoke(
        [
            SystemMessage(
                content=(
                    "You extract restaurant orders. The only menu items are Burger, Pizza, "
                    "Momos, and Malai_chaap. Return invalid for unrelated requests. "
                    "Return accept_partial or decline when the user is answering a prior "
                    "partial-availability question."
                )
            ),
            HumanMessage(content=text),
        ]
    )
    return result if isinstance(result, OrderRequest) else OrderRequest.model_validate(result)


def _llm_node(state: OrderState) -> dict:
    text = state.get("user_input", "").strip()
    if not text:
        return {"status": "invalid", "response": "Please enter a dish and quantity."}

    request = extract_order(text)
    updates: dict = {"messages": [HumanMessage(content=text)]}
    if request.action == "accept_partial":
        if state.get("status") != "partial":
            return {
                **updates,
                "status": "invalid",
                "response": "There is no partial order to accept.",
            }
        return {
            **updates,
            "required_quantity": int(state["available_quantity"]),
            "status": "confirmed",
            "response": "I will prepare the available quantity.",
        }
    if request.action == "decline":
        return {
            **updates,
            "status": "failed",
            "final_result": "Order cancelled.",
            "response": "Order cancelled.",
        }
    if request.action != "new_order" or not request.dish or request.quantity is None:
        return {
            **updates,
            "status": "invalid",
            "response": "I am a food-ordering assistant. Please order a menu item with a quantity.",
        }

    return {
        **updates,
        "dish": _normalise_dish(request.dish) or request.dish,
        "required_quantity": request.quantity,
        "order_attempts": state.get("order_attempts", 0) + 1,
        "status": "new",
    }


def _order_confirm_node(state: OrderState) -> dict:
    dish = state["dish"]
    requested = state["required_quantity"]
    available = MENU.get(dish, 0)
    if available == 0:
        return {
            "available_quantity": "0",
            "status": "unavailable",
            "response": f"Sorry, {dish} is not available. Please choose another order.",
        }
    if requested > available:
        return {
            "available_quantity": str(available),
            "status": "partial",
            "response": (
                f"Only {available} {dish} are available, but you requested {requested}. "
                "Reply 'yes' to accept the partial order or place a new order."
            ),
        }
    return {"available_quantity": str(available), "status": "confirmed"}


def _cook_node(
    state: OrderState,
    probability: Callable[[], float] = random.random,
) -> dict:
    attempts_left = state.get("cook_attempts", 2)
    if probability() < 0.4:
        attempts_left -= 1
        if attempts_left <= 0:
            return {
                "cook_attempts": 0,
                "status": "failed",
                "final_result": "We are sorry, but we could not cook your order.",
                "response": "We are sorry, but we could not cook your order.",
            }
        return {
            "cook_attempts": attempts_left,
            "status": "confirmed",
            "response": "Cooking failed. Retrying once.",
        }
    return {"cook_attempts": attempts_left, "status": "ready"}


def _serve_node(
    state: OrderState,
    probability: Callable[[], float] = random.random,
) -> dict:
    attempts_left = state.get("serve_attempts", 2)
    if probability() < 0.4:
        attempts_left -= 1
        if attempts_left <= 0:
            return {
                "serve_attempts": 0,
                "status": "failed",
                "final_result": "We are sorry, but we could not serve your order.",
                "response": "We are sorry, but we could not serve your order.",
            }
        return {
            "serve_attempts": attempts_left,
            "status": "ready",
            "response": "Serving failed. Retrying once.",
        }
    return {"serve_attempts": attempts_left, "status": "complete"}


def _finalize_node(state: OrderState) -> dict:
    message = (
        f"Your order of {state['required_quantity']} {state['dish']} is complete."
        if state.get("status") == "complete"
        else state.get("final_result", state.get("response", "Order ended."))
    )
    return {"final_result": message, "response": message}


def _route_after_llm(state: OrderState) -> str:
    status = state.get("status")
    if status == "new":
        return "order_confirm"
    if status == "confirmed":
        return "cook"
    return "finish" if status in {"invalid", "failed"} else "finish"


def _route_after_confirm(state: OrderState) -> str:
    if state["status"] == "confirmed":
        return "cook"
    if state.get("order_attempts", 0) >= 3:
        return "finish"
    return "wait_for_user"


def _route_after_cook(state: OrderState) -> str:
    if state["status"] == "ready":
        return "serve"
    if state["status"] == "confirmed":
        return "cook"
    return "finish"


def _route_after_serve(state: OrderState) -> str:
    if state["status"] == "complete":
        return "finish"
    if state["status"] == "ready":
        return "cook"
    return "finish"


def create_order_graph(
    probability: Callable[[], float] = random.random,
):
    builder = StateGraph(OrderState)
    builder.add_node("llm", _llm_node)
    builder.add_node("order_confirm", _order_confirm_node)
    builder.add_node("cook", lambda state: _cook_node(state, probability))
    builder.add_node("serve", lambda state: _serve_node(state, probability))
    builder.add_node("wait_for_user", lambda _: {})
    builder.add_node("finish", _finalize_node)
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", _route_after_llm)
    builder.add_conditional_edges("order_confirm", _route_after_confirm)
    builder.add_conditional_edges("cook", _route_after_cook)
    builder.add_conditional_edges("serve", _route_after_serve)
    builder.add_edge("wait_for_user", END)
    builder.add_edge("finish", END)
    return builder.compile()


def run_cli() -> None:
    graph = create_order_graph()
    state: OrderState = {
        "messages": [],
        "order_attempts": 0,
        "cook_attempts": 2,
        "serve_attempts": 2,
    }
    print("Restaurant ordering assistant. Type 'quit' to exit.")
    while True:
        text = input("You: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        state = graph.invoke({**state, "user_input": text})
        if state.get("response"):
            print(f"Assistant: {state['response']}")
        if state.get("status") in {"complete", "failed"}:
            break


if __name__ == "__main__":
    run_cli()
