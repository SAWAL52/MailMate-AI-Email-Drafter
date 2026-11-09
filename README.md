# MailMate – Automatic Email Drafter

A small Streamlit app that creates email drafts with Groq and LangGraph. It follows a human-in-the-loop workflow: generate a draft, review it, request revisions, and approve the final version.

## Features

- Generate an email from a recipient, purpose, tone, and additional details
- Review and edit the generated draft
- Ask the AI to revise the draft using feedback
- Approve the final version
- Start a fresh email whenever needed

**Note:** The app prepares email text only. It does not connect to an inbox or send emails.

## Setup

Python 3.10 or newer is recommended.

1. Clone the repository and open its folder.
2. Create and activate a virtual environment.
3. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and add your Groq API key:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

## Workflow

The LangGraph workflow has three nodes:

- `draft`: generates the first email or revises it using feedback
- `human_review`: pauses for the user's review using a LangGraph interrupt
- `final`: stores the approved draft as the final response

The graph uses an in-memory checkpointer. Conversation state is kept while the Streamlit process is running; it is not a permanent database.

## Tech stack

- Python
- Streamlit
- LangChain Groq
- LangGraph
- Pydantic is not required for the app's state definition
- python-dotenv
