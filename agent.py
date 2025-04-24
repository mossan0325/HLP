# -*- coding: utf-8 -*-
"""
Agent workflow: Patient  ↔  Nurse  ↔  Doctor-GPT
All prompts are now in ENGLISH.
"""
import os, operator
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ─────────────────── Patient data ───────────────────
patient_data = {
    "diagnosis": "Stomach cancer",          # ground-truth diagnosis (not revealed)
    "interview": {
        "chief_complaint": "Epigastric pain, early satiety, and recent weight loss.",
        "history_of_present_illness": "Symptoms began 3 months ago with mild indigestion, then worsening pain and loss of appetite.",
        "past_medical_history": "Gastric ulcer 10 years ago; otherwise healthy.",
        "allergies": "Penicillin – causes rash. No food allergies.",
        "current_medications": "Occasional OTC antacids and daily vitamin supplements."
    },
    "hidden_info": {
        "family_history": "Father died of stomach cancer at age 58.",
        "alcohol_use": "Binge drinks on weekends.",
        "H_pylori_infection": "Tested positive 2 years ago but never treated.",
        "black_stool": "Has noticed tarry stools recently but hasn’t reported it.",
        "anemia_symptoms": "Often feels tired and dizzy but assumes it’s stress."
    }
}

# ─────────────────── Typed state ───────────────────
class ConsultationState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    initial_complaint: str
    nurse_summary: Optional[str]
    doctor_analysis: Optional[str]
    follow_up_questions: Optional[List[str]]
    conclusion: Optional[str]
    patient_data: Dict[str, Any]

# ─────────────────── Patient simulator ───────────────────
def simulate_patient_response(
    questions: List[str],
    history: List[AnyMessage],
    patient_data: Dict[str, Any]
) -> str:
    """Return patient’s answers to the nurse’s / doctor’s questions."""
    patient_context = f"""
You are the PATIENT with the following information.

**Information you will volunteer without being asked**
- Chief complaint: {patient_data['interview']['chief_complaint']}
- History of present illness: {patient_data['interview']['history_of_present_illness']}
- Past medical history: {patient_data['interview']['past_medical_history']}
- Allergies: {patient_data['interview']['allergies']}
- Current medications: {patient_data['interview']['current_medications']}

**Information you prefer to keep private but WILL reveal if directly asked**
- Family history: {patient_data['hidden_info']['family_history']}
- Alcohol use: {patient_data['hidden_info']['alcohol_use']}
- H. pylori infection: {patient_data['hidden_info']['H_pylori_infection']}
- Stool details: {patient_data['hidden_info']['black_stool']}
- Anemia-like symptoms: {patient_data['hidden_info']['anemia_symptoms']}

Answer the following questions naturally as the patient. Reveal hidden info ONLY if a question explicitly targets it.
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", patient_context),
        MessagesPlaceholder("history"),
        ("human", "Please answer these questions:\n- " + "\n- ".join(questions)),
    ])
    return (prompt | llm).invoke({"history": history}).content

# ─────────────────── Nurse: initial intake ───────────────────
def run_nurse_initial_intake(state: ConsultationState) -> ConsultationState:
    print("--- Nurse: initial intake ---")
    history, pdata = state["messages"], state["patient_data"]
    initial_cc = pdata["interview"]["chief_complaint"]
    new_msgs: List[AnyMessage] = []

    mandatory_qs = [
        "What specific symptoms are you experiencing right now?",
        "When did these symptoms start and how have they changed?",
        "Do you have any significant past illnesses or medical conditions?",
        "Do you have any allergies to medications or food?",
        "Are you currently taking any prescription, OTC drugs, or supplements?"
    ]

    new_msgs.append(AIMessage(content="Hello doctor."))
    new_msgs.append(HumanMessage(
        content="Hi, I’m the nurse. I’d like to ask you a few basic questions about your symptoms."
    ))

    answers = simulate_patient_response(mandatory_qs, history + new_msgs, pdata)
    new_msgs.append(AIMessage(content=answers))

    summary = (
        "Initial intake summary:\n"
        f"- Chief complaint: {initial_cc}\n"
        "- Answers to mandatory questions:\n"
        f"{answers}"
    )
    return {
        "messages": new_msgs,
        "nurse_summary": summary,
        "initial_complaint": initial_cc
    }

# ─────────────────── Nurse: follow-up ───────────────────
def run_nurse_ask_followup(state: ConsultationState) -> ConsultationState:
    print("--- Nurse: follow-up questions ---")
    if not state.get("follow_up_questions"):
        return {"messages": []}

    qs = state["follow_up_questions"]
    new_msgs = [HumanMessage(
        content="The doctor has a few follow-up questions:\n- " + "\n- ".join(qs)
    )]
    answer = simulate_patient_response(qs, state["messages"] + new_msgs, state["patient_data"])
    new_msgs.append(AIMessage(content=answer))
    return {"messages": new_msgs, "follow_up_questions": None}

# ─────────────────── Doctor ───────────────────
def run_doctor_analysis(state: ConsultationState) -> ConsultationState:
    print("--- Doctor: analysis ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an experienced physician. Analyze the entire conversation so far "
         "(including nurse intake). Your goal is to identify the most likely cause of the patient’s problem.\n\n"
         "If you need MORE information, respond with:\n"
         "CONTINUE\n- question1\n- question2 ...\n\n"
         "If you can form a preliminary diagnosis / plan WITHOUT more questions, respond with:\n"
         "CONCLUDE\nYour reasoning, differential diagnosis, suggested work-up, next steps."),
        MessagesPlaceholder("history"),
    ])
    result = (prompt | llm).invoke({"history": state["messages"]}).content.strip()

    new_msgs: List[AnyMessage] = []
    if result.startswith("CONTINUE"):
        qs = [l.lstrip("- ").strip() for l in result.splitlines()[1:] if l.strip()]
        new_msgs.append(SystemMessage(content=f"[Doctor note] Needs follow-up: {qs}"))
        return {"messages": new_msgs, "doctor_analysis": result,
                "follow_up_questions": qs, "conclusion": None}

    if result.startswith("CONCLUDE"):
        conclusion = result[len("CONCLUDE"):].strip()
        new_msgs.append(SystemMessage(content=f"[Doctor conclusion]\n{conclusion}"))
        return {"messages": new_msgs, "doctor_analysis": result,
                "follow_up_questions": None, "conclusion": conclusion}

    # Fallback
    new_msgs.append(SystemMessage(content=f"[Doctor conclusion – unexpected format]\n{result}"))
    return {"messages": new_msgs, "doctor_analysis": result,
            "follow_up_questions": None, "conclusion": result}

# ─────────────────── Routing logic ───────────────────
def should_continue(state: ConsultationState) -> str:
    return "ask_followup" if state.get("follow_up_questions") else END

# ─────────────────── Build graph ───────────────────
workflow = StateGraph(ConsultationState)
workflow.add_node("nurse_initial_intake", run_nurse_initial_intake)
workflow.add_node("run_doctor_analysis", run_doctor_analysis)
workflow.add_node("nurse_followup", run_nurse_ask_followup)

workflow.set_entry_point("nurse_initial_intake")
workflow.add_edge("nurse_initial_intake", "run_doctor_analysis")
workflow.add_conditional_edges("run_doctor_analysis", should_continue,
                               {"ask_followup": "nurse_followup", END: END})
workflow.add_edge("nurse_followup", "run_doctor_analysis")
app = workflow.compile()

if __name__ == "__main__":
    init_state = {"messages": [], "patient_data": patient_data}
    final = app.invoke(init_state, {"recursion_limit": 10})
    print("\n=== Conversation ===")
    for m in final["messages"]:
        print(f"{m.__class__.__name__}: {m.content}")
    print("\nDoctor’s final conclusion:\n", final.get("conclusion", "None"))
