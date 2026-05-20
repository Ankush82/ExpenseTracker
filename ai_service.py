"""OpenRouter AI calls: categorisation, image extraction, financial advice."""

import base64
import json
import re
from typing import Optional  # noqa: F401 — used in type hints

from openai import OpenAI

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Model used for text tasks (fast + cheap)
TEXT_MODEL = "anthropic/claude-3.5-haiku-20241022"
# Model used for vision and long-form advice
VISION_MODEL = "anthropic/claude-3.5-sonnet-20241022"

CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Sports & Fitness",
    "Travel & Transport",
    "Shopping",
    "Entertainment",
    "Bills & EMI",
    "Healthcare",
    "Utilities",
    "Personal Care",
    "Indulgence",
    "Other",
]


def _client(api_key: str) -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE, api_key=api_key)


def _extract_json(text: str) -> dict:
    """Pull first JSON object from a model response."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError("No JSON found in response")


# ---------------------------------------------------------------------------
# Categorisation
# ---------------------------------------------------------------------------

def categorize_transaction(
    api_key: str,
    merchant: str,
    amount: float,
    description: str = "",
    bank: str = "",
) -> str:
    """Return one of the CATEGORIES for a given transaction."""
    prompt = f"""You are an expense categoriser for Indian users.
Classify this bank transaction into exactly one of these categories:
{', '.join(CATEGORIES)}

Transaction:
- Merchant / Description: {merchant or description}
- Amount: INR {amount:,.2f}
- Bank info: {bank}

Reply with ONLY the category name, nothing else."""

    try:
        resp = _client(api_key).chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
        )
        raw = resp.choices[0].message.content.strip()
        # Exact match first
        if raw in CATEGORIES:
            return raw
        # Partial match
        for cat in CATEGORIES:
            if cat.lower() in raw.lower() or raw.lower() in cat.lower():
                return cat
        return "Other"
    except Exception:
        return "Other"


def categorize_item(api_key: str, item_name: str) -> str:
    """Categorise a single line item from a receipt."""
    prompt = f"""Categorise this item into one of: {', '.join(CATEGORIES)}
Item: {item_name}
Reply with ONLY the category name."""
    try:
        resp = _client(api_key).chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
        )
        raw = resp.choices[0].message.content.strip()
        if raw in CATEGORIES:
            return raw
        for cat in CATEGORIES:
            if cat.lower() in raw.lower():
                return cat
        return "Other"
    except Exception:
        return "Other"


# ---------------------------------------------------------------------------
# Image / screenshot extraction
# ---------------------------------------------------------------------------

def extract_from_screenshot(api_key: str, image_bytes: bytes, mime: str = "image/jpeg") -> dict:
    """
    Send a payment screenshot to the vision model.
    Returns:
    {
        "merchant": str,
        "total_amount": float,
        "date": str | null,
        "items": [{"name": str, "quantity": str, "price": float}]
    }
    """
    b64 = base64.b64encode(image_bytes).decode()

    prompt = """Analyse this payment / order screenshot and extract the following as JSON:
{
  "merchant": "<app or store name>",
  "total_amount": <final amount paid as a number>,
  "date": "<YYYY-MM-DD if visible, else null>",
  "items": [
    {"name": "<item name>", "quantity": "<e.g. 2x or 180ml x 3>", "price": <price as number>}
  ]
}

If the screenshot is from a delivery app like Blinkit, Swiggy, Zomato, Amazon etc., list every line item.
Use the bill total / final amount for total_amount (after discounts).
Respond with ONLY valid JSON, no markdown, no explanation."""

    try:
        resp = _client(api_key).chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=800,
        )
        return _extract_json(resp.choices[0].message.content)
    except Exception as e:
        return {
            "merchant": "Unknown",
            "total_amount": 0.0,
            "date": None,
            "items": [],
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Financial advice
# ---------------------------------------------------------------------------

def get_financial_advice(
    api_key: str,
    spending_summary: str,
    user_name: str,
    period_label: str = "this month",
) -> str:
    prompt = f"""You are a friendly personal finance advisor helping Indian users save money.

{user_name}'s spending summary for {period_label}:
{spending_summary}

Provide a concise analysis with these sections:
1. **Overall Assessment** (2-3 sentences on overall spending health)
2. **Top Spending Areas** (call out the biggest categories and whether they seem reasonable)
3. **3 Actionable Tips** to reduce spending, specific to the data above
4. **Estimated Monthly Savings Potential** if tips are followed

Be warm, specific, and non-judgmental. Use INR amounts where relevant.
Keep the total response under 400 words."""

    try:
        resp = _client(api_key).chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Could not generate advice: {e}"
