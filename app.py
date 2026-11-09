import os
import re
import smtplib
import uuid
from email.message import EmailMessage
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
st.caption("Draft, review, and send emails with AI and your Gmail account.")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to your .env file and restart the app.")
    st.stop()


class MailState(TypedDict):
    sender: str
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

Sender: {state['sender']}
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

Sender: {state['sender']}
Recipient: {state['recipient']}
Purpose: {state['purpose']}
Tone: {state['tone']}
Additional details: {state['details']}

Requirements:
- Start with a suitable subject line in the format: Subject: ...
- Use a greeting and a clear, concise body.
- End with a suitable closing and the sender's name only if provided; otherwise use [Your Name].
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
            return {"draft": review.get("edited_draft", state["draft"]), "human_feedback": ""}
        return {
            "draft": review.get("edited_draft", state["draft"]),
            "human_feedback": review.get("feedback", "").strip()
                or "Improve the email while keeping its original meaning.",
        }

    def route_after_review(state: MailState):
        return "draft" if state.get("human_feedback") else "final"

    def finalize(state: MailState):
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


def split_subject_and_body(draft: str):
    match = re.search(r"(?im)^\s*subject\s*:\s*(.+?)\s*$", draft)
    if match:
        subject = match.group(1).strip()
        body = draft[:match.start()] + draft[match.end():]
        body = body.strip()
        return subject, body
    return "Message from MailMate", draft.strip()


def send_gmail(sender: str, recipient: str, draft: str):
    gmail_address = os.getenv("GMAIL_ADDRESS", "").strip()
    app_password = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "")

    if not gmail_address or not app_password:
        raise ValueError(
            "Gmail credentials are missing. Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD in your .env file."
        )
    if sender.strip().lower() != gmail_address.lower():
        raise ValueError(
            f"The From address must match the configured Gmail account ({gmail_address})."
        )

    subject, body = split_subject_and_body(draft)
    message = EmailMessage()
    message["From"] = gmail_address
    message["To"] = recipient.strip()
    message["Subject"] = subject
    message.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
        smtp.login(gmail_address, app_password)
        smtp.send_message(message)


graph = build_graph()

def get_interrupt(result):
    interrupts = result.get("__interrupt__", [])
    return interrupts[0].value if interrupts else None


for key, value in {
    "thread_id": str(uuid.uuid4()),
    "stage": "input",
    "draft": "",
    "final_email": "",
    "sender": os.getenv("GMAIL_ADDRESS", "example@gmail.com"),
    "recipient": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.subheader("Workflow")
    st.write("1. Enter sender and recipient.")
    st.write("2. Generate an email draft.")
    st.write("3. Review, edit, or request changes.")
    st.write("4. Approve and explicitly send.")
    st.divider()
    st.caption("Emails are sent only when you click Send email.")
    if st.button("Start a new email", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.stage = "input"
        st.session_state.draft = ""
        st.session_state.final_email = ""
        st.rerun()

if st.session_state.stage == "input":
    with st.form("email_request"):
        sender = st.text_input("From", value=st.session_state.sender, placeholder="example@gmail.com")
        recipient = st.text_input("To", placeholder="recipient@example.com")
        purpose = st.text_input("What is this email about?", placeholder="e.g. Request for a meeting")
        tone = st.selectbox("Tone", ["Professional", "Formal", "Friendly", "Polite", "Concise"])
        details = st.text_area(
            "Additional details",
            placeholder="Include relevant dates, names, background, requests, and important points.",
            height=150,
        )
        submitted = st.form_submit_button("Generate email", type="primary", use_container_width=True)

    if submitted:
        if not sender.strip() or not recipient.strip() or not purpose.strip():
            st.warning("Please enter the From address, To address, and email purpose.")
        elif "@" not in sender or "@" not in recipient:
            st.warning("Please enter valid-looking email addresses.")
        else:
            st.session_state.sender = sender.strip()
            st.session_state.recipient = recipient.strip()
            initial_state = {
                "sender": sender.strip(),
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
    st.caption(f"From: {st.session_state.sender}  ·  To: {st.session_state.recipient}")
    edited_draft = st.text_area("Email draft", value=st.session_state.draft, height=320)
    feedback = st.text_area(
        "Request changes (optional)",
        placeholder="e.g. Make it shorter or sound more polite.",
        height=90,
    )
    col1, col2 = st.columns(2)
    with col1:
        approve = st.button("Approve draft", type="primary", use_container_width=True)
    with col2:
        revise = st.button("Revise draft", use_container_width=True)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    if approve:
        result = graph.invoke(
            Command(resume={"decision": "approve", "edited_draft": edited_draft, "feedback": ""}),
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
            st.session_state.draft = get_interrupt(result)["draft"]
            st.rerun()

elif st.session_state.stage == "final":
    st.success("Draft approved. Review it once more before sending.")
    st.caption(f"From: {st.session_state.sender}  ·  To: {st.session_state.recipient}")
    final_email = st.text_area("Final email", value=st.session_state.final_email, height=320)
    st.session_state.final_email = final_email

    if st.button("✉️ Send email", type="primary", use_container_width=True):
        try:
            send_gmail(st.session_state.sender, st.session_state.recipient, final_email)
            st.success(f"Email successfully sent to {st.session_state.recipient}.")
            st.session_state.stage = "sent"
            st.rerun()
        except Exception as exc:
            st.error(f"Could not send email: {exc}")

elif st.session_state.stage == "sent":
    st.success(f"Email sent successfully to {st.session_state.recipient}.")
    st.button(
        "Compose another email",
        type="primary",
        on_click=lambda: st.session_state.update({
            "thread_id": str(uuid.uuid4()),
            "stage": "input",
            "draft": "",
            "final_email": "",
            "recipient": "",
        }),
    )
