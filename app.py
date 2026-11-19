import os
import re
import smtplib
import uuid
from datetime import datetime
from email.message import EmailMessage
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

st.set_page_config(page_title="MailMate | AI Email Drafter", page_icon="✉️", layout="centered")
st.title("✉️ MailMate AI")
st.caption("Draft, review, and send intelligent emails with LangGraph & your Gmail account.")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to your .env file and restart the app.")
    st.stop()

DEFAULT_SUBJECT = "Message from MailMate"
TONES = ["Professional", "Formal", "Friendly", "Polite", "Concise", "Persuasive"]
EMAIL_RE = re.compile(r"^[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+$")


def parse_addresses(raw: str) -> list[str]:
    """Split a comma / semicolon / newline separated string into unique addresses."""
    seen, result = set(), []
    for part in re.split(r"[,;\n]+", raw or ""):
        addr = part.strip()
        if addr and addr.lower() not in seen:
            seen.add(addr.lower())
            result.append(addr)
    return result


def invalid_addresses(addresses: list[str]) -> list[str]:
    return [a for a in addresses if not EMAIL_RE.match(a)]


def split_subject_and_body(text: str, fallback_subject: str = "") -> tuple[str, str]:
    """Split the AI output into (subject, body). Expects a first 'Subject: ...' line."""
    match = re.search(r"(?im)^\s*\**subject\**\s*:\s*\**(.+?)\**\s*$", text)
    if match:
        subject = match.group(1).strip()
        body = (text[:match.start()] + text[match.end():]).strip()
        return subject or fallback_subject or DEFAULT_SUBJECT, body
    return fallback_subject or DEFAULT_SUBJECT, text.strip()


def fmt(addresses_raw: str) -> str:
    items = parse_addresses(addresses_raw)
    return ", ".join(items) if items else "—"

class MailState(TypedDict):
    sender: str
    recipient: str
    cc: str
    bcc: str
    subject: str
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

Current subject: {state['subject']}

Current draft body:
{state['draft']}

User feedback:
{state['human_feedback']}

Rules:
- Keep the requested tone: {state['tone']}.
- Also revise the subject line so it matches the revised email and the user's feedback. Keep it short, clear and specific (under 10 words), and make it a genuinely updated subject rather than a copy of the current one, unless the current subject already fits perfectly.
- Output format: first line "Subject: <subject>", then a blank line, then the complete revised email body.
- Do not invent facts or add unsupported details.
"""
        else:
            if state.get("subject"):
                subject_rule = f"Use this exact subject: {state['subject']}"
            else:
                subject_rule = "Suggest a short, clear, specific subject line (under 10 words)."
            prompt = f"""
Write an email based on the information below.

Sender: {state['sender']}
Recipient: {state['recipient']}
Purpose: {state['purpose']}
Tone: {state['tone']}
Additional details: {state['details']}

Requirements:
- {subject_rule}
- Write in a {state['tone']} tone.
- Use a greeting and a clear, concise body that covers the purpose and details.
- End with a suitable closing and the sender's name only if provided; otherwise use [Your Name].
- Do not invent dates, names, reasons, or other facts.
- Output format: first line "Subject: <subject>", then a blank line, then the email body. Nothing else.
"""
        response = llm.invoke(prompt)
        subject, body = split_subject_and_body(response.content, state.get("subject", ""))
        return {"subject": subject, "draft": body, "human_feedback": ""}

    def finalize(state: MailState):
        return {"final_response": state["draft"]}

    builder = StateGraph(MailState)
    builder.add_node("draft", draft_email)
    builder.add_node("final", finalize)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "final")
    builder.add_edge("final", END)

    return builder.compile(checkpointer=InMemorySaver())


def send_gmail(sender: str, to: list[str], cc: list[str], bcc: list[str], subject: str, body: str):
    gmail_address = os.getenv("GMAIL_ADDRESS", "").strip()
    app_password = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "")

    if not gmail_address or not app_password:
        raise ValueError("Gmail credentials are missing. Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD in your .env file.")

    if sender.strip().lower() != gmail_address.lower():
        raise ValueError(f"The From address must match the configured Gmail account ({gmail_address}).")

    if not to:
        raise ValueError("At least one recipient is required in the To field.")

    message = EmailMessage()
    message["From"] = gmail_address
    message["To"] = ", ".join(to)
    if cc:
        message["Cc"] = ", ".join(cc)
    # BCC addresses are deliberately NOT added as a header, so other recipients
    # never see them. They are only passed to the SMTP envelope below.
    message["Subject"] = subject.strip() or DEFAULT_SUBJECT
    message.set_content(body)

    envelope_recipients, seen = [], set()
    for addr in to + cc + bcc:
        if addr.lower() not in seen:
            seen.add(addr.lower())
            envelope_recipients.append(addr)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
        smtp.login(gmail_address, app_password)
        smtp.send_message(message, from_addr=gmail_address, to_addrs=envelope_recipients)


graph = build_graph()


def fresh_state() -> dict:
    return {
        "thread_id": str(uuid.uuid4()),
        "stage": "input",
        "draft": "",
        "recipient": "",
        "cc": "",
        "bcc": "",
        "subject": "",
        "draft_history": [],
        "purpose": "",
        "tone": "Professional",
        "details": "",
    }


defaults = {**fresh_state(), "sender": os.getenv("GMAIL_ADDRESS", "example@gmail.com"), "email_history": []}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def graph_state(human_feedback: str = "", draft: str = "") -> dict:
    ss = st.session_state
    return {
        "sender": ss.sender,
        "recipient": ss.recipient,
        "cc": ss.cc,
        "bcc": ss.bcc,
        "subject": ss.subject,
        "purpose": ss.purpose,
        "tone": ss.tone,
        "details": ss.details,
        "draft": draft,
        "human_feedback": human_feedback,
        "final_response": "",
    }


def add_version(subject: str, draft: str):
    """Remember every AI-generated draft so earlier ones can be loaded again."""
    history = st.session_state.draft_history
    if history and history[-1]["subject"] == subject and history[-1]["draft"] == draft:
        return
    history.append({
        "label": f"Version {len(history) + 1}" + (" (initial)" if not history else ""),
        "subject": subject,
        "draft": draft,
    })


with st.sidebar:
    st.subheader("Workflow Steps")
    st.markdown(
        "1. Input details & recipients\n"
        "2. AI draft generation\n"
        "3. Review & feedback\n"
        "4. Final preview\n"
        "5. Send via Gmail SMTP"
    )
    st.divider()

    if st.button("✨ Start New Email", use_container_width=True):
        st.session_state.update(fresh_state())
        st.rerun()

    if st.session_state.email_history:
        st.subheader("📦 Session Outbox")
        for item in st.session_state.email_history[-5:]:
            st.caption(f"To: {item['recipient']} ({item['time']})")

config = {"configurable": {"thread_id": st.session_state.thread_id}}

if st.session_state.stage == "input":
    with st.form("email_request"):
        col1, col2 = st.columns(2)
        with col1:
            sender = st.text_input("From", value=st.session_state.sender)
        with col2:
            recipient = st.text_input(
                "To",
                value=st.session_state.recipient,
                placeholder="name@example.com, other@example.com",
            )

        cc = st.text_input(
            "CC (optional)",
            value=st.session_state.cc,
            placeholder="Separate multiple addresses with commas",
        )
        bcc = st.text_input(
            "BCC (optional)",
            value=st.session_state.bcc,
            placeholder="Hidden from all other recipients; separate with commas",
        )
        subject = st.text_input(
            "Subject (optional)",
            value=st.session_state.subject,
            placeholder="Leave blank and the AI will suggest one — you can edit it later",
        )

        purpose = st.text_input(
            "What is this email about?",
            value=st.session_state.purpose,
            placeholder="e.g., Requesting project extension approval",
        )
        tone = st.selectbox("Tone", TONES, index=TONES.index(st.session_state.tone))
        details = st.text_area(
            "Additional details / context",
            value=st.session_state.details,
            placeholder="Include relevant background points, metrics, or timeline constraints.",
            height=130,
        )
        submitted = st.form_submit_button("⚡ Generate AI Draft", type="primary", use_container_width=True)

    if submitted:
        to_list = parse_addresses(recipient)
        cc_list = parse_addresses(cc)
        bcc_list = parse_addresses(bcc)
        bad = invalid_addresses([sender.strip()] if sender.strip() else []) \
            + invalid_addresses(to_list) + invalid_addresses(cc_list) + invalid_addresses(bcc_list)

        if not sender.strip() or not to_list or not purpose.strip():
            st.warning("Please fill out the From, To, and Purpose fields.")
        elif bad:
            st.warning("These email addresses look invalid: " + ", ".join(bad))
        else:
            st.session_state.sender = sender.strip()
            st.session_state.recipient = ", ".join(to_list)
            st.session_state.cc = ", ".join(cc_list)
            st.session_state.bcc = ", ".join(bcc_list)
            st.session_state.subject = subject.strip()
            st.session_state.purpose = purpose.strip()
            st.session_state.tone = tone
            st.session_state.details = details.strip()

            with st.spinner("Drafting your email with Groq AI..."):
                result = graph.invoke(graph_state(), config=config)
                st.session_state.draft = result.get("draft", "")
                st.session_state.subject = result.get("subject") or DEFAULT_SUBJECT
                st.session_state.draft_history = []
                add_version(st.session_state.subject, st.session_state.draft)
                st.session_state.stage = "review"
                st.rerun()

elif st.session_state.stage == "review":
    st.subheader("📝 Review & Refine Draft")
    st.caption(
        f"To: {fmt(st.session_state.recipient)} | CC: {fmt(st.session_state.cc)} | "
        f"BCC: {fmt(st.session_state.bcc)} | Tone: {st.session_state.tone}"
    )

    history = st.session_state.draft_history
    if len(history) > 1:
        with st.expander(f"🕘 Load previous AI draft ({len(history)} versions)"):
            idx = st.selectbox(
                "Version",
                list(range(len(history))),
                index=len(history) - 1,
                format_func=lambda i: f"{history[i]['label']} — {history[i]['subject']}",
            )
            st.text_area(
                "Selected version",
                value=history[idx]["draft"],
                height=200,
                disabled=True,
                key=f"history_preview_{idx}_{len(history)}",
            )
            if st.button("↩️ Load this version", use_container_width=True):
                st.session_state.subject = history[idx]["subject"]
                st.session_state.draft = history[idx]["draft"]
                st.rerun()

    edited_subject = st.text_input(
        "Subject (AI-suggested — edit if you like)",
        value=st.session_state.subject or DEFAULT_SUBJECT,
    )
    edited_draft = st.text_area("Message", value=st.session_state.draft, height=280)
    feedback = st.text_area(
        "Feedback for AI Revision (Optional)",
        placeholder="e.g., Make the second paragraph punchier and more concise.",
        height=80,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        approve = st.button("✅ Continue to Preview", type="primary", use_container_width=True)
    with col2:
        revise = st.button("🔄 Request Revision", use_container_width=True)
    with col3:
        back_to_input = st.button("← Edit Recipients", use_container_width=True)

    if back_to_input:
        st.session_state.subject = edited_subject.strip() or DEFAULT_SUBJECT
        st.session_state.draft = edited_draft
        st.session_state.stage = "input"
        st.rerun()

    if approve:
        if not edited_draft.strip():
            st.warning("The message body is empty.")
        else:
            st.session_state.subject = edited_subject.strip() or DEFAULT_SUBJECT
            st.session_state.draft = edited_draft
            st.session_state.stage = "final"
            st.rerun()

    if revise:
        if not feedback.strip():
            st.warning("Please enter feedback notes for the AI to update the draft.")
        else:
            st.session_state.subject = edited_subject.strip() or DEFAULT_SUBJECT
            with st.spinner("Refining draft..."):
                result = graph.invoke(
                    graph_state(human_feedback=feedback.strip(), draft=edited_draft),
                    config=config,
                )
                st.session_state.draft = result.get("draft", edited_draft)
                st.session_state.subject = result.get("subject") or st.session_state.subject
                add_version(st.session_state.subject, st.session_state.draft)
                st.rerun()

elif st.session_state.stage == "final":
    st.subheader("🔍 Final Preview")
    st.info("Please check everything below. The email will only be sent when you click **Send email**.")

    with st.container(border=True):
        st.markdown(f"**From:** {st.session_state.sender}")
        st.markdown(f"**To:** {fmt(st.session_state.recipient)}")
        st.markdown(f"**CC:** {fmt(st.session_state.cc)}")
        st.markdown(f"**BCC:** {fmt(st.session_state.bcc)}  \n*(hidden from other recipients)*")
        st.markdown(f"**Subject:** {st.session_state.subject}")
        st.markdown("**Message:**")
        st.text_area(
            "Message",
            value=st.session_state.draft,
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )

    col1, col2 = st.columns(2)
    with col1:
        send = st.button("🚀 Send email", type="primary", use_container_width=True)
    with col2:
        back = st.button("← Back to edit", use_container_width=True)

    if back:
        st.session_state.stage = "review"
        st.rerun()

    if send:
        try:
            with st.spinner("Connecting to SMTP server..."):
                send_gmail(
                    st.session_state.sender,
                    parse_addresses(st.session_state.recipient),
                    parse_addresses(st.session_state.cc),
                    parse_addresses(st.session_state.bcc),
                    st.session_state.subject,
                    st.session_state.draft,
                )
            st.session_state.email_history.append(
                {"recipient": st.session_state.recipient, "time": datetime.now().strftime("%H:%M:%S")}
            )
            st.session_state.stage = "sent"
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to transmit email: {exc}")


elif st.session_state.stage == "sent":
    st.balloons()
    st.success(f"Successfully sent email to **{st.session_state.recipient}**.")
    if st.button("Compose Another Email", type="primary", use_container_width=True):
        st.session_state.update(fresh_state())
        st.rerun()