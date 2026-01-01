# Enhancing Nurse-Led Initial Patient Intake with an LLM Physician Agent

This repository hosts a course/workshop-style paper:

**“Enhancing Nurse-Led Initial Patient Intake with an LLM Physician Agent:
A Detailed Evaluation Using a Simulation Scenario”**

## Summary
Initial patient intake by nurses can be limited by time constraints and variable clinical experience, which may reduce follow-up questioning and risk missing serious conditions.  
This project proposes a lightweight protocol that inserts an LLM (GPT-4o) as an **on-demand virtual physician (“Doctor-GPT”)** into the nurse interview loop to generate follow-up questions and support differential diagnosis in real time.

## Method (High-level)
A simulation-based evaluation compares:
- **Baseline:** nurse-only intake
- **Intervention:** nurse + Doctor-GPT loop (iterative follow-up)

System components (implemented with a multi-agent workflow):
- Patient Simulator (reveals hidden details only if asked)
- Nurse Agent (asks mandatory intake questions and summarizes)
- Doctor-GPT Agent (suggests follow-ups or concludes with differential + work-up)

## Key Findings (from one gastric cancer scenario)
- The Intervention generated additional follow-up questions, elicited hidden findings (e.g., tarry stools, binge drinking),
  included the true diagnosis (**gastric cancer**) in the differential, and proposed concrete work-up plans (e.g., EGD, H. pylori test, CBC).
- The Baseline focused on ulcer recurrence, uncovered no hidden info, and provided no test recommendations.

## Contents
- `Enhancing Nurse-Led Initial Patient Intake with an LLM Physician Agent.pdf` : Paper (main artifact)

## Limitations
This is a qualitative, single-case simulation study; future work includes multi-case evaluation, real-world validation, and safety assessment.
