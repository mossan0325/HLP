from langgraph.graph import StateGraph, Node, Message
from openai import OpenAI
import json, os, tiktoken, copy

openai = OpenAI()
MODEL = "gpt-4o-mini"

# ---------- State ----------
INIT_STATE = {
    "persona": None,          # dict
    "core_info": {},          # filled by NurseCore
    "follow_info": {},        # filled by NurseFollowUp
    "ratings":   {},          # at the end
    "transcript": []          # list[dict(role,content)]
}

# ---------- Helper ----------
def call_gpt(messages, **params):
    return openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        **params
    ).choices[0].message.content

# ---------- Nodes ----------
class NurseCore(Node):
    def run(self, state):
        persona = state["persona"]
        needed = ["chief_complaint","onset_course","severity_scale",
                  "associated_symptoms","relevant_history","impact_on_daily_life"]
        for key in needed:
            if key not in state["core_info"]:
                q = f"{key.replace('_',' ')} について教えてください。"
                state["transcript"].append({"role":"assistant","content":q})
                ans = call_gpt([{"role":"user","content":persona['profile']['chief_complaint']},
                                *state["transcript"][-1:]])
                state["transcript"].append({"role":"user","content":ans})
                state["core_info"][key] = ans
                break
        if len(state["core_info"]) == 6:
            state["core_done"] = True
        return state

class PhysicianReasoner(Node):
    SYS = "You are an experienced physician..."
    def run(self, state):
        if state.get("core_done") is not True:
            return state   # skip until core done
        transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state["transcript"]])
        prompt = [{"role":"system","content":self.SYS},
                  {"role":"user","content":transcript}]
        reply = call_gpt(prompt)
        try:
            data = json.loads(reply)
        except:
            data = {"action":"END"}
        state["phys_output"] = data
        return state

class NurseFollowUp(Node):
    def run(self, state):
        data = state.get("phys_output",{})
        if data.get("action")=="END":
            state["follow_done"] = True
            return state
        question = data["question"]
        state["transcript"].append({"role":"assistant","content":question})
        ans = call_gpt([*state["transcript"][-1:]])
        state["transcript"].append({"role":"user","content":ans})
        state["follow_info"][question] = ans
        return state

class NurseCloser(Node):
    def run(self, state):
        if not state.get("follow_done",False):
            return state
        summary = f"まとめますと…{state['core_info']|state['follow_info']}"
        ask = "面接は以上です。Comfort / Stress / Satisfaction を1–5で教えてください。"
        state["transcript"].append({"role":"assistant","content":summary+"\n"+ask})
        rating = call_gpt(state["transcript"][-1:])
        state["ratings"] = rating
        state["done"] = True
        return state

# ---------- Build Graph ----------
g = StateGraph(initial_state=INIT_STATE)
g.add_node("nurse_core", NurseCore())
g.add_node("physician", PhysicianReasoner())
g.add_node("nurse_followup", NurseFollowUp())
g.add_node("nurse_closer", NurseCloser())

g.add_edge("nurse_core","physician")
g.add_edge("physician","nurse_followup")
g.add_edge("nurse_followup","physician")  # loop
g.add_edge("nurse_followup","nurse_closer")
g.add_edge("nurse_closer","END")

graph = g.compile()
