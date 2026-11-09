# MailMate – AI Email Drafter

MailMate is a simple AI-powered email drafting application built with Python and Streamlit. It uses Groq to generate email drafts and LangGraph to manage the drafting, review, and revision process.

Instead of sending an email immediately after generation, MailMate allows users to review and edit the draft before approving and sending it through Gmail.

## Features

- **AI Email Generation:** Generate email drafts using the Groq LLM API.
- **Multiple Writing Tones:** Choose between Professional, Formal, Friendly, Polite, and Concise.
- **Human-in-the-Loop Review:** Review the generated email before sending.
- **Email Revision:** Provide feedback and ask the AI to improve the draft.
- **Manual Editing:** Edit the generated email directly.
- **Gmail Integration:** Send approved emails using Gmail SMTP.
- **Secure Credentials:** Store API keys and Gmail credentials in environment variables.

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq API
- LangGraph
- Gmail SMTP
- Python-dotenv

## Project Structure

```text
MailMate-AI-Email-Drafter/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/MailMate-AI-Email-Drafter.git
cd MailMate-AI-Email-Drafter
```

### 2. Create a virtual environment

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory and add:

```env
GROQ_API_KEY=your_groq_api_key
GMAIL_ADDRESS=your_email@gmail.com
GMAIL_APP_PASSWORD=your_gmail_app_password
```

You can obtain a Groq API key from [Groq Console](https://console.groq.com/).

### 5. Configure Gmail

MailMate uses Gmail SMTP to send emails.

To enable Gmail sending:

1. Enable 2-Step Verification on your Google Account.
2. Generate an App Password if your account is eligible.
3. Add your Gmail address to `GMAIL_ADDRESS`.
4. Add the generated App Password to `GMAIL_APP_PASSWORD`.

**Important:** Use a Google App Password, not your regular Gmail account password. Keep your `.env` file private and never upload it to GitHub.

### 6. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## How It Works

MailMate follows a human-in-the-loop workflow:

1. **Enter Email Details:** Provide the sender address, recipient address, email purpose, preferred tone, and additional information.
2. **Generate Draft:** Groq generates an email based on the provided details.
3. **Review Draft:** LangGraph pauses the workflow for human review.
4. **Revise or Approve:** Edit the draft directly or provide feedback for another AI-generated version.
5. **Send Email:** Once satisfied, approve the draft and explicitly click the send button.
6. **Gmail SMTP:** The application sends the approved email using the configured Gmail account.

## Workflow Architecture

```text
User Input
    |
    v
Generate Email (Groq)
    |
    v
Human Review (LangGraph Interrupt)
    |
    +------ Revise ------> Generate Email Again
    |                            |
    |                            v
    |                       Human Review
    |
    +------ Approve
              |
              v
        Final Email Draft
              |
              v
       User Clicks Send
              |
              v
          Gmail SMTP
              |
              v
         Email Sent
```

## Security Notes

- Never commit your `.env` file or expose API keys.
- Use Gmail App Passwords instead of your regular account password.
- The sender address must match the configured `GMAIL_ADDRESS`.
- Emails are not sent automatically after generation. User approval and an explicit send action are required.
- Review the email recipient and content before sending.

## Future Improvements

- Support for multiple email providers.
- Email templates for common use cases.
- Conversation history and saved drafts.
- Support for email attachments.
- More advanced personalization options.

