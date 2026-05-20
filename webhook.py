"""
FastAPI webhook server — receives SMS from iPhone Shortcut.
Runs on port 8000 alongside Streamlit (port 8501).
"""

from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database as db
import sms_parser

app = FastAPI(title="Expense Tracker Webhook", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

db.init_db()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _verify_token(token: Optional[str]):
    expected = db.get_config("webhook_token")
    if not token or token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Token header")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------



class SMSPayload(BaseModel):
    message: str
    user_id: int = 1  # Which of the 3 users sent this SMS


class ManualTransaction(BaseModel):
    user_id: int
    amount: float
    merchant: str
    date: Optional[str] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "expense-tracker-webhook"}


@app.post("/sms")
def receive_sms(payload: SMSPayload, x_token: Optional[str] = Header(None)):
    """
    Called by iPhone Shortcut whenever a bank SMS arrives.
    Parses the message and saves it as a pending transaction.
    """
    _verify_token(x_token)

    text = payload.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message")

    parsed = sms_parser.parse_sms(text)

    if not parsed:
        # Not a debit SMS — ignore silently so the Shortcut doesn't show an error
        return {"status": "skipped", "reason": "Not a recognised debit SMS"}

    tx_id = db.save_transaction(
        user_id=payload.user_id,
        amount=parsed["amount"],
        merchant=parsed["merchant"],
        date=parsed["date"],
        description=parsed.get("description", parsed["merchant"]),
        category="Uncategorised",
        source="shortcut",
        raw_text=parsed["raw_text"],
        is_pending=True,
    )

    return {
        "status": "saved",
        "transaction_id": tx_id,
        "amount": parsed["amount"],
        "merchant": parsed["merchant"],
        "date": parsed["date"],
        "bank": parsed.get("bank", ""),
    }


@app.get("/pending")
def list_pending(user_id: Optional[int] = None, x_token: Optional[str] = Header(None)):
    """List pending (unreviewed) transactions."""
    _verify_token(x_token)
    df = db.get_pending_transactions(user_id)
    return df.to_dict(orient="records")


@app.get("/token")
def get_token(x_token: Optional[str] = Header(None)):
    """Return the current webhook token (used for admin verification)."""
    _verify_token(x_token)
    return {"token": db.get_config("webhook_token")}
