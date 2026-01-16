import os
import json
import pandas as pd
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from google import genai

# =========================
# CONFIG
# =========================

EXCEL_PATH = "tasks.xlsx"

client = genai.Client(
    api_key="Your API Key"
)

# =========================
# STATE
# =========================

class TaskState(TypedDict):
    excel_path: str
    current_row: int
    task: dict
    plan: dict
    result: str
    status: str
    has_more: bool

# =========================
# LLM HELPER
# =========================

def chat(prompt: str) -> str:
    print(prompt)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text.strip()

# =========================
# LOAD NEXT TASK
# =========================

def load_task(state: TaskState):
    df = pd.read_excel(state["excel_path"])

    pending = df[df["status"].isna() | (df["status"] != "done")]

    if pending.empty:
        return {"has_more": False}

    row_index = pending.index[0]
    task = df.loc[row_index].to_dict()

    return {
        "current_row": row_index,
        "task": task,
        "has_more": True
    }

# =========================
# INTENT INTERPRETER (BRAIN)
# =========================

def interpret_intent(state: TaskState):
    text = state["task"]["task_input"]

    prompt = f"""
You are an intent understanding agent.

Understand the user's intent and choose ONE action.

Available actions:
- make_folder → user wants to create a directory or folder
- search → user wants information, research, or summary
- question → user asks a direct question
- unknown → unclear intent

Return ONLY valid JSON.

JSON format:
{{
  "action": "make_folder | search | question | unknown",
  "reason": "short explanation"
}}

User input:
{text}
"""

    raw = chat(prompt)

    try:
        plan = json.loads(raw)
    except Exception:
        plan = {"action": "unknown", "reason": "Invalid JSON from LLM"}

    return {"plan": plan}

# =========================
# ROUTER
# =========================

def route_action(state: TaskState):
    return state["plan"]["action"]

# =========================
# ACTION NODES
# =========================

def make_folder(state: TaskState):
    try:
        path = state["task"]["task_input"]
        os.makedirs(path, exist_ok=True)
        return {
            "result": f"Folder created successfully: {path}",
            "status": "done"
        }
    except Exception as e:
        return {
            "result": str(e),
            "status": "failed"
        }

def search_task(state: TaskState):
    query = state["task"]["task_input"]
    result = chat(f"Search and summarize:\n{query}")
    return {"result": result, "status": "done"}

def question_task(state: TaskState):
    question = state["task"]["task_input"]
    result = chat(question)
    return {"result": result, "status": "done"}

def unknown_task(state: TaskState):
    return {
        "result": "Could not understand the intent",
        "status": "failed"
    }

# =========================
# UPDATE EXCEL
# =========================

def update_excel(state: TaskState):
    df = pd.read_excel(state["excel_path"])

    df.at[state["current_row"], "status"] = state["status"]
    df.at[state["current_row"], "result"] = state["result"]

    df.to_excel(state["excel_path"], index=False)

    return {}

# =========================
# BUILD LANGGRAPH
# =========================

builder = StateGraph(TaskState)

builder.add_node("load_task", load_task)
builder.add_node("interpret_intent", interpret_intent)

builder.add_node("make_folder", make_folder)
builder.add_node("search", search_task)
builder.add_node("question", question_task)
builder.add_node("unknown", unknown_task)

builder.add_node("update_excel", update_excel)

builder.set_entry_point("load_task")

builder.add_conditional_edges(
    "load_task",
    lambda s: "interpret_intent" if s["has_more"] else "end",
    {
        "interpret_intent": "interpret_intent",
        "end": END
    }
)

builder.add_conditional_edges(
    "interpret_intent",
    route_action,
    {
        "make_folder": "make_folder",
        "search": "search",
        "question": "question",
        "unknown": "unknown"
    }
)

builder.add_edge("make_folder", "update_excel")
builder.add_edge("search", "update_excel")
builder.add_edge("question", "update_excel")
builder.add_edge("unknown", "update_excel")

builder.add_edge("update_excel", "load_task")

app = builder.compile()

# =========================
# RUN
# =========================

if __name__ == "__main__":
    print("🚀 Starting agentic Excel task executor...")
    app.invoke({"excel_path": EXCEL_PATH})
    print("✅ All tasks processed.")
