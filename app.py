import os
import uuid
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

st.set_page_config(page_title="MailMate | Email Drafter", page_icon="✉️", layout="centered")

st.title("✉️ MailMate")
st.caption("Draft professional emails with AI, review them, and request changes before using them.")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to your .env file and restart the app.")
    st.stop()


class MailState(TypedDict):
    recipient: str
    purpose: str
    tone: str
    details: str
    draft: str
    human_feedback: str
    final_response: str


@st.cache_resource
def build_graph():
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    def draft_email(state: MailState):
        if state.get("human_feedback"):
            prompt = f"""
Revise the email using the user's feedback.

Original request:
Recipient: {state['recipient']}
Purpose: {state['purpose']}
Tone: {state['tone']}
Additional details: {state['details']}

Current draft:
{state['draft']}

User feedback:
{state['human_feedback']}

Return the complete revised email only. Do not invent facts or add unsupported details.
"""
        else:
            prompt = f"""
Write a complete email based on the information below.

Recipient: {state['recipient']}
Purpose: {state['purpose']}
Tone: {state['tone']}
Additional details: {state['details']}

Requirements:
- Include a suitable subject line.
- Use a greeting and a clear, concise body.
- End with a suitable closing and a placeholder for the sender's name if it was not provided.
- Do not invent dates, names, reasons, or other facts.
- Return only the email draft.
"""
        response = llm.invoke(prompt)
        return {"draft": response.content, "human_feedback": ""}

    def human_review(state: MailState):
        review = interrupt({
            "draft": state["draft"],
            "question": "Review the draft. Approve it or provide feedback for a revision."
        })

        if review.get("decision") == "approve":
            return {
                "draft": review.get("edited_draft", state["draft"]),
                "human_feedback": "",
            }

        edited_draft = review.get("edited_draft", state["draft"])
        feedback = review.get("feedback", "").strip()
        return {
            "draft": edited_draft,
            "human_feedback": feedback or "Improve the email while keeping its original meaning.",
        }

    def route_after_review(state: MailState):
        return "draft" if state.get("human_feedback") else "final"

    def finalize(state: MailState):
        # This app prepares a final draft; it does not send email.
        return {"final_response": state["draft"]}

    builder = StateGraph(MailState)
    builder.add_node("draft", draft_email)
    builder.add_node("human_review", human_review)
    builder.add_node("final", finalize)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "human_review")
    builder.add_conditional_edges("human_review", route_after_review)
    builder.add_edge("final", END)

    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()


def get_interrupt(result):
    interrupts = result.get("__interrupt__", [])
    return interrupts[0].value if interrupts else None


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "stage" not in st.session_state:
    st.session_state.stage = "input"
if "draft" not in st.session_state:
    st.session_state.draft = ""
if "final_email" not in st.session_state:
    st.session_state.final_email = ""


with st.sidebar:
    st.subheader("How it works")
    st.write("1. Describe the email you need.")
    st.write("2. AI creates a first draft.")
    st.write("3. Review, edit, approve, or request changes.")
    st.write("4. Copy the final email.")
    st.divider()
    st.caption("Your approval is required before a draft is finalized. No email is sent automatically.")
    if st.button("Start a new email", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.stage = "input"
        st.session_state.draft = ""
        st.session_state.final_email = ""
        st.rerun()


if st.session_state.stage == "input":
    with st.form("email_request"):
        recipient = st.text_input("To", placeholder="e.g. The College Principal")
        purpose = st.text_input("What is this email about?", placeholder="e.g. Request for medical leave")
        tone = st.selectbox("Tone", ["Professional", "Formal", "Friendly", "Polite", "Concise"])
        details = st.text_area(
            "Additional details",
            placeholder="Include relevant dates, names, background, requests, and any points that must be mentioned.",
            height=160,
        )
        submitted = st.form_submit_button("Generate email", type="primary", use_container_width=True)

    if submitted:
        if not recipient.strip() or not purpose.strip():
            st.warning("Please enter the recipient and the purpose of the email.")
        else:
            initial_state = {
                "recipient": recipient.strip(),
                "purpose": purpose.strip(),
                "tone": tone,
                "details": details.strip(),
                "draft": "",
                "human_feedback": "",
                "final_response": "",
            }
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = graph.invoke(initial_state, config=config)
            st.session_state.draft = get_interrupt(result)["draft"]
            st.session_state.stage = "review"
            st.rerun()


elif st.session_state.stage == "review":
    st.subheader("Review your email")
    edited_draft = st.text_area("Email draft", value=st.session_state.draft, height=320)
    feedback = st.text_area(
        "Request changes (optional)",
        placeholder="e.g. Make it shorter, sound more polite, or mention that I have attached the document.",
        height=100,
    )

    col1, col2 = st.columns(2)
    with col1:
        approve = st.button("Approve draft", type="primary", use_container_width=True)
    with col2:
        revise = st.button("Revise draft", use_container_width=True)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    if approve:
        result = graph.invoke(
            Command(resume={
                "decision": "approve",
                "edited_draft": edited_draft,
                "feedback": "",
            }),
            config=config,
        )
        st.session_state.final_email = result.get("final_response", edited_draft)
        st.session_state.stage = "final"
        st.rerun()

    if revise:
        if not feedback.strip():
            st.warning("Please describe what you want changed.")
        else:
            result = graph.invoke(
                Command(resume={
                    "decision": "revise",
                    "edited_draft": edited_draft,
                    "feedback": feedback.strip(),
                }),
                config=config,
            )
            interrupt_data = get_interrupt(result)
            st.session_state.draft = interrupt_data["draft"]
            st.rerun()


elif st.session_state.stage == "final":
    st.success("Your email draft is approved.")
    st.text_area("Final email", value=st.session_state.final_email, height=340)
    st.caption("This app does not send the email. Copy the approved draft into your email client.")
    st.button(
        "Create another email",
        type="primary",
        on_click=lambda: (
            st.session_state.update({
                "thread_id": str(uuid.uuid4()),
                "stage": "input",
                "draft": "",
                "final_email": "",
            })
        ),
    )
