# -*- coding: utf-8 -*-
import os, operator
from typing import TypedDict, Annotated, List, Optional, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AnyMessage, SystemMessage, HumanMessage, AIMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

# ─────────────────── Environment & LLM ───────────────────
load_dotenv()                # reads your .env file for OPENAI_API_KEY
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ─────────────────── Hard-coded patient case ───────────────────
patient_data = {
    "diagnosis": "Stomach cancer",          # ground truth (hidden)
    "interview": {
        "chief_complaint": "Epigastric pain, early satiety, and recent weight loss.",
        "history_of_present_illness": (
            "Symptoms began about 3 months ago with mild indigestion, then "
            "worsening pain and loss of appetite."
        ),
        "past_medical_history": "Gastric ulcer 10 years ago; otherwise healthy.",
        "allergies": "Penicillin – causes rash. No food allergies.",
        "current_medications": (
            "Occasional over-the-counter antacids and daily vitamin supplements."
        )
    },
    "hidden_info": {
        "family_history": "Father died of stomach cancer at age 58.",
        "alcohol_use": "Binge drinks on weekends.",
        "H_pylori_infection": "Tested positive 2 years ago but never treated.",
        "black_stool": "Has noticed tarry stools recently but hasn’t reported it.",
        "anemia_symptoms": "Often feels tired and dizzy but assumes it’s stress."
    }
}

# ─────────────────── State definition ───────────────────
class ConsultationState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    initial_complaint: str
    nurse_summary: Optional[str]
    conclusion: Optional[str]
    patient_data: Dict[str, Any]

# ─────────────────── Patient simulator ───────────────────
def simulate_patient_response(
    questions: List[str],
    history: List[AnyMessage],
    pdata: Dict[str, Any]
) -> str:
    """Generate answers *as* the patient, using hidden rules."""
    patient_ctx = f"""
You are the PATIENT with these details.

**Info you volunteer freely**
- Chief complaint: {pdata['interview']['chief_complaint']}
- HPI: {pdata['interview']['history_of_present_illness']}
- PMH: {pdata['interview']['past_medical_history']}
- Allergies: {pdata['interview']['allergies']}
- Current meds: {pdata['interview']['current_medications']}

**Info you reveal ONLY if explicitly asked**
- Family history: {pdata['hidden_info']['family_history']}
- Alcohol use: {pdata['hidden_info']['alcohol_use']}
- H. pylori history: {pdata['hidden_info']['H_pylori_infection']}
- Stool details: {pdata['hidden_info']['black_stool']}
- Anemia-like symptoms: {pdata['hidden_info']['anemia_symptoms']}

Answer the incoming questions naturally as this patient. If a question
does NOT clearly target a hidden point, do NOT reveal it.
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", patient_ctx),
        MessagesPlaceholder("history"),
        ("human", "Please answer:\n- " + "\n- ".join(questions))
    ])
    return (prompt | llm).invoke({"history": history}).content.strip()

# ─────────────────── Nurse: mandatory intake ───────────────────
def run_nurse_initial_intake(state: ConsultationState) -> ConsultationState:
    print("--- Nurse: initial intake ---")
    history, pdata = state["messages"], state["patient_data"]
    cc = pdata["interview"]["chief_complaint"]
    new_msgs: List[AnyMessage] = []

    mandatory_qs = [
        "What specific symptoms are you experiencing right now?",
        "When did these symptoms start and how have they changed?",
        "Do you have any significant past illnesses or medical conditions?",
        "Do you have any allergies to medications or food?",
        "Are you currently taking any prescription, over-the-counter drugs, or supplements?"
    ]

    # (1) patient greeting
    new_msgs.append(AIMessage(content="Hello."))
    # (2) nurse begins intake
    new_msgs.append(HumanMessage(
        content="Hi, I’m the nurse. Let’s start with a few basic questions."
    ))
    # (3) patient answers mandatory questions
    answers = simulate_patient_response(mandatory_qs, history + new_msgs, pdata)
    new_msgs.append(AIMessage(content=answers))

    summary = (
        "Intake summary:\n"
        f"- Chief complaint: {cc}\n"
        "- Answers to mandatory questions:\n"
        f"{answers}"
    )
    return {
        "messages": new_msgs,
        "nurse_summary": summary,
        "initial_complaint": cc
    }

# ─────────────────── Nurse: simple diagnosis ───────────────────
def run_nurse_analysis(state: ConsultationState) -> ConsultationState:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a junior nurse with limited clinical knowledge. "
         "Based ONLY on the conversation so far, list up to three possible "
         "diagnoses and pick the single most likely one. "
         "Prefix your answer with 'CONCLUDE'. "
         "Do NOT ask the patient any more questions."),
        MessagesPlaceholder("history")
    ])
    out = (prompt | llm).invoke({"history": state["messages"]}).content.strip()
    conclusion = out.lstrip("CONCLUDE").strip()
    return {
        "messages": [SystemMessage(content=f"[Nurse conclusion]\n{conclusion}")],
        "conclusion": conclusion
    }

# ─────────────────── Graph wiring ───────────────────
baseline = StateGraph(ConsultationState)
baseline.add_node("nurse_intake", run_nurse_initial_intake)
baseline.add_node("nurse_analysis", run_nurse_analysis)
baseline.set_entry_point("nurse_intake")
baseline.add_edge("nurse_intake", "nurse_analysis")
baseline_app = baseline.compile()

# ─────────────────── Execute ───────────────────
if __name__ == "__main__":
    init_state = {"messages": [], "patient_data": patient_data}
    final_state = baseline_app.invoke(init_state, {"recursion_limit": 10})

    print("\n=== Conversation ===")
    for msg in final_state["messages"]:
        role = msg.__class__.__name__
        print(f"{role}: {msg.content}")
    print("\nNurse’s final conclusion:\n", final_state.get("conclusion", "None"))
