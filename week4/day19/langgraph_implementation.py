from typing import TypedDict
from langgraph.graph import END, StateGraph

class State(TypedDict):
    number: int

def double_number(state: State) -> dict:
    boardnum=state["number"]
    newnum=boardnum*2
    print("in double_number node")
    print(newnum)
    return {"number": newnum}

def finish(state: State)->dict:
    boardnum=state["number"]
    print("in finish node")
    print(boardnum)
    return {"number": boardnum}

def decision(state: State) -> str:
    print("in decision node which is not a state node")
    if state["number"] < 100:
        return "double_number"
    else:
        return "finish"

    
builder=StateGraph(State)
builder.add_node("double_number", double_number)
builder.add_node("finish", finish)

#entry point of the graph
builder.set_entry_point("double_number")

#add decision node
builder.add_conditional_edges("double_number", decision, {"double_number": "double_number", "finish": "finish"})

builder.add_edge("finish", END)

graph=builder.compile()


if __name__ == "__main__":
    result=graph.invoke({"number": 5})