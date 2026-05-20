"""Parse Indian bank debit SMS messages into structured transaction data."""

import re
from datetime import datetime
from dateutil import parser as dateutil_parser
from typing import Optional


# Month abbreviation map for Indian bank SMS formats
_MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}


def _parse_amount(s: str) -> float:
    """Convert '12,200.00' or '2095' to float."""
    return float(s.replace(",", ""))


def _normalize_date(date_str: str) -> Optional[str]:
    """Return YYYY-MM-DD from various Indian bank date formats."""
    date_str = date_str.strip().upper()

    # DD-MM-YY  e.g. 19-05-26
    m = re.match(r"^(\d{2})-(\d{2})-(\d{2})$", date_str)
    if m:
        day, month, year = m.groups()
        full_year = f"20{year}"
        return f"{full_year}-{month}-{day}"

    # DD-MMM-YY  e.g. 20-MAY-26
    m = re.match(r"^(\d{2})-([A-Z]{3})-(\d{2})$", date_str)
    if m:
        day, mon, year = m.groups()
        month = _MONTH_MAP.get(mon, "01")
        full_year = f"20{year}"
        return f"{full_year}-{month}-{day}"

    # DD-MMM-YYYY  e.g. 20-May-2026
    m = re.match(r"^(\d{2})-([A-Za-z]{3})-(\d{4})$", date_str)
    if m:
        day, mon, year = m.groups()
        month = _MONTH_MAP.get(mon.upper(), "01")
        return f"{year}-{month}-{day}"

    # DD/MM/YYYY
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", date_str)
    if m:
        day, month, year = m.groups()
        return f"{year}-{month}-{day}"

    # Try dateutil as fallback
    try:
        return dateutil_parser.parse(date_str).strftime("%Y-%m-%d")
    except Exception:
        return datetime.today().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Bank-specific patterns
# Each pattern is a list of (regex, extractor_function) pairs.
# Extractor receives match object and returns dict with keys:
#   amount (float), merchant (str), date (str YYYY-MM-DD), description (str), bank (str)
# ---------------------------------------------------------------------------

def _try_axis_spent(text: str) -> Optional[dict]:
    """
    Spent INR 2095 Axis Bank Card no. XX4830 19-05-26 09:17:50 IST BUNDL TECHN Avl Limit: INR 196122.5
    """
    m = re.search(
        r"Spent INR ([\d,]+(?:\.\d+)?)\s+Axis Bank.*?(\d{2}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}\s+IST\s+(.+?)\s+Avl",
        text, re.IGNORECASE
    )
    if m:
        return {
            "amount": _parse_amount(m.group(1)),
            "date": _normalize_date(m.group(2)),
            "merchant": m.group(3).strip(),
            "description": m.group(3).strip(),
            "bank": "Axis Bank",
        }
    return None


def _try_hdfc_debited(text: str) -> Optional[dict]:
    """
    UPDATE: INR 12,200.00 debited from HDFC Bank XX7750 on 20-MAY-26. Info: ACH D- BOI EMI COLLECTION-....
    """
    m = re.search(
        r"INR ([\d,]+(?:\.\d+)?)\s+debited from HDFC Bank\s+\S+\s+on\s+(\d{2}-[A-Za-z]{3}-\d{2,4})\.\s+Info:\s*(.+?)(?:\.|Avl)",
        text, re.IGNORECASE
    )
    if m:
        desc = m.group(3).strip()
        return {
            "amount": _parse_amount(m.group(1)),
            "date": _normalize_date(m.group(2)),
            "merchant": desc,
            "description": desc,
            "bank": "HDFC Bank",
        }
    return None


def _try_hdfc_alert(text: str) -> Optional[dict]:
    """
    HDFC Bank: Rs.500 debited from a/c XX1234 on 20-05-26 for UPI-...
    """
    m = re.search(
        r"HDFC Bank.*?Rs\.?\s*([\d,]+(?:\.\d+)?)\s+debited.*?on\s+(\d{2}-\d{2}-\d{2,4})\s+for\s+(.+?)(?:\.|$)",
        text, re.IGNORECASE
    )
    if m:
        desc = m.group(3).strip()
        return {
            "amount": _parse_amount(m.group(1)),
            "date": _normalize_date(m.group(2)),
            "merchant": desc,
            "description": desc,
            "bank": "HDFC Bank",
        }
    return None


def _try_sbi(text: str) -> Optional[dict]:
    """
    Your A/c XX1234 is debited by Rs.500.00 on 20-05-26 trf to MERCHANT Ref No 123
    """
    m = re.search(
        r"debited by Rs\.?\s*([\d,]+(?:\.\d+)?)\s+on\s+(\d{2}-\d{2}-\d{2,4})\s+trf\s+to\s+(.+?)(?:Ref|$)",
        text, re.IGNORECASE
    )
    if m:
        desc = m.group(3).strip()
        return {
            "amount": _parse_amount(m.group(1)),
            "date": _normalize_date(m.group(2)),
            "merchant": desc,
            "description": desc,
            "bank": "SBI",
        }
    return None


def _try_sbi_upi(text: str) -> Optional[dict]:
    """
    SBI A/c XX1234 debited INR 500.00 on 20-05-2026 & credited to VPA merchant@upi
    """
    m = re.search(
        r"SBI.*?debited INR ([\d,]+(?:\.\d+)?)\s+on\s+(\d{2}-\d{2}-\d{4})\s+.*?credited to(?:\s+VPA)?\s+(.+?)(?:\s|$)",
        text, re.IGNORECASE
    )
    if m:
        desc = m.group(3).strip()
        return {
            "amount": _parse_amount(m.group(1)),
            "date": _normalize_date(m.group(2)),
            "merchant": desc,
            "description": desc,
            "bank": "SBI",
        }
    return None


def _try_icici(text: str) -> Optional[dict]:
    """
    Handles multiple ICICI formats:
    1. ICICI Bank Acct XX791 debited for Rs 502.99 on 20-May-26; HEISETASSE BEVE credited.
    2. ICICI Bank Acct XX1234 debited for Rs 500.00 on 20-May-2026; ... trf to MERCHANT
    """
    # Extract amount and date first
    m = re.search(
        r"ICICI Bank.*?debited for Rs\.?\s*([\d,]+(?:\.\d+)?)\s+on\s+(\d{2}-[A-Za-z]+-\d{2,4})",
        text, re.IGNORECASE
    )
    if not m:
        return None

    amount = _parse_amount(m.group(1))
    date = _normalize_date(m.group(2))
    rest = text[m.end():]

    # Try: "; MERCHANT credited" or "; MERCHANT UPI"
    merchant_m = re.search(r";\s*(.+?)\s+(?:credited|UPI:|debited|Call)", rest, re.IGNORECASE)
    if not merchant_m:
        # Try: "trf to MERCHANT" or "at MERCHANT"
        merchant_m = re.search(r"(?:trf to|at)\s+(.+?)(?:;|\.|$)", rest, re.IGNORECASE)
    if not merchant_m:
        # Use whatever comes after the semicolon
        merchant_m = re.search(r";\s*([A-Z][A-Za-z0-9 &\-_]{2,40})", rest)

    merchant = merchant_m.group(1).strip() if merchant_m else "ICICI Transaction"

    return {
        "amount": amount,
        "date": date,
        "merchant": merchant,
        "description": merchant,
        "bank": "ICICI Bank",
    }


def _try_kotak(text: str) -> Optional[dict]:
    """
    Kotak Bk: INR 500.00 debited from AC XXXXXX1234 on 20-05-2026 to MERCHANT
    """
    m = re.search(
        r"Kotak.*?INR ([\d,]+(?:\.\d+)?)\s+debited.*?on\s+(\d{2}-\d{2}-\d{4})\s+to\s+(.+?)(?:\.|$)",
        text, re.IGNORECASE
    )
    if m:
        desc = m.group(3).strip()
        return {
            "amount": _parse_amount(m.group(1)),
            "date": _normalize_date(m.group(2)),
            "merchant": desc,
            "description": desc,
            "bank": "Kotak Bank",
        }
    return None


def _try_generic_inr_debited(text: str) -> Optional[dict]:
    """
    Fallback: any 'INR X debited' or 'debited INR X' pattern
    """
    # Pattern: INR amount debited ... on date ... for/at merchant
    m = re.search(
        r"INR ([\d,]+(?:\.\d+)?)\s+debited.*?on\s+(\d{2}-[A-Za-z\d]+-\d{2,4})",
        text, re.IGNORECASE
    )
    if m:
        # Try to extract merchant from "for ..." or "to ..."
        merchant_m = re.search(r"(?:for|to|at)\s+([A-Z][A-Z\s\-_@.]{2,40})", text[m.end():])
        merchant = merchant_m.group(1).strip() if merchant_m else "Unknown"
        return {
            "amount": _parse_amount(m.group(1)),
            "date": _normalize_date(m.group(2)),
            "merchant": merchant,
            "description": text[:120],
            "bank": "Unknown",
        }

    # Pattern: debited for Rs X  (ICICI / generic)
    m = re.search(
        r"debited\s+for\s+Rs\.?\s*([\d,]+(?:\.\d+)?)\s+on\s+(\d{2}-[A-Za-z]+-\d{2,4})",
        text, re.IGNORECASE
    )
    if m:
        date = _normalize_date(m.group(2))
        rest = text[m.end():]
        merchant_m = re.search(r";\s*(.+?)\s+(?:credited|UPI:|Call)", rest, re.IGNORECASE)
        merchant = merchant_m.group(1).strip() if merchant_m else "Unknown"
        return {
            "amount": _parse_amount(m.group(1)),
            "date": date,
            "merchant": merchant,
            "description": text[:120],
            "bank": "Unknown",
        }

    # Pattern: debited Rs/INR X (loose)
    m = re.search(
        r"debited\s+(?:Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)",
        text, re.IGNORECASE
    )
    if m:
        date_m = re.search(r"(\d{2}[-/]\d{2}[-/]\d{2,4}|\d{2}-[A-Za-z]{3}-\d{2,4})", text)
        date = _normalize_date(date_m.group(1)) if date_m else datetime.today().strftime("%Y-%m-%d")
        merchant_m = re.search(r"(?:for|to|at|UPI-)\s*([A-Za-z][\w\s\-@.]{2,40})", text)
        merchant = merchant_m.group(1).strip() if merchant_m else "Unknown"
        return {
            "amount": _parse_amount(m.group(1)),
            "date": date,
            "merchant": merchant,
            "description": text[:120],
            "bank": "Unknown",
        }

    return None


def _try_upi_mandate(text: str) -> Optional[dict]:
    """
    UPI Mandate / UPI payment multi-line format:
      UPI Mandate:
      Sent Rs.179.00
      from HDFC Bank A/c 7750
      To Spotify India Pvt Ltd
      19/04/26
    Also handles single-line variants like:
      UPI: Sent Rs.500 to Merchant on 20/05/26
    """
    # Collapse multi-line to single line for easier regex
    flat = " ".join(text.split())

    # Multi-line UPI Mandate: Sent Rs.X ... To MERCHANT ... DD/MM/YY
    m = re.search(
        r"Sent Rs\.?\s*([\d,]+(?:\.\d+)?)\s+from\s+.+?To\s+(.+?)\s+(\d{2}[/-]\d{2}[/-]\d{2,4})",
        flat, re.IGNORECASE
    )
    if m:
        return {
            "amount": _parse_amount(m.group(1)),
            "merchant": m.group(2).strip(),
            "description": m.group(2).strip(),
            "date": _normalize_date(m.group(3).replace("/", "-")),
            "bank": _extract_bank(flat),
        }

    # Sent Rs.X to MERCHANT on DATE
    m = re.search(
        r"Sent Rs\.?\s*([\d,]+(?:\.\d+)?)\s+to\s+(.+?)\s+(?:on\s+)?(\d{2}[/-]\d{2}[/-]\d{2,4})",
        flat, re.IGNORECASE
    )
    if m:
        return {
            "amount": _parse_amount(m.group(1)),
            "merchant": m.group(2).strip(),
            "description": m.group(2).strip(),
            "date": _normalize_date(m.group(3).replace("/", "-")),
            "bank": _extract_bank(flat),
        }

    # Sent Rs.X to MERCHANT (no explicit date — use today)
    m = re.search(
        r"Sent Rs\.?\s*([\d,]+(?:\.\d+)?)\s+to\s+(.+?)(?:\s+Ref|\s+Not You|\s+$)",
        flat, re.IGNORECASE
    )
    if m:
        date_m = re.search(r"(\d{2}[/-]\d{2}[/-]\d{2,4})", flat)
        date = _normalize_date(date_m.group(1).replace("/", "-")) if date_m else datetime.today().strftime("%Y-%m-%d")
        return {
            "amount": _parse_amount(m.group(1)),
            "merchant": m.group(2).strip(),
            "description": m.group(2).strip(),
            "date": date,
            "bank": _extract_bank(flat),
        }

    return None


def _try_upi_generic(text: str) -> Optional[dict]:
    """
    Generic UPI payment messages:
      Rs.X paid to MERCHANT via UPI on DD-MM-YYYY
      You have paid Rs X to MERCHANT on DD/MM/YY
      Payment of Rs X to MERCHANT successful
    """
    flat = " ".join(text.split())

    patterns = [
        r"(?:paid|transferred|debited)\s+(?:Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)\s+(?:to|for)\s+(.+?)\s+(?:via|on|through|Ref)",
        r"You have paid\s+(?:Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)\s+to\s+(.+?)\s+on",
        r"Payment of\s+(?:Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)\s+to\s+(.+?)\s+(?:successful|done|complete)",
    ]
    for pat in patterns:
        m = re.search(pat, flat, re.IGNORECASE)
        if m:
            date_m = re.search(r"(\d{2}[/-]\d{2}[/-]\d{2,4}|\d{2}-[A-Za-z]{3}-\d{2,4})", flat)
            date = _normalize_date(date_m.group(1).replace("/", "-")) if date_m else datetime.today().strftime("%Y-%m-%d")
            return {
                "amount": _parse_amount(m.group(1)),
                "merchant": m.group(2).strip(),
                "description": m.group(2).strip(),
                "date": date,
                "bank": _extract_bank(flat),
            }
    return None


def _extract_bank(text: str) -> str:
    """Pull bank name from any SMS text."""
    for bank in ["HDFC Bank", "Axis Bank", "SBI", "ICICI Bank", "Kotak Bank",
                 "Yes Bank", "IDFC Bank", "IndusInd Bank", "Bank of Baroda", "PNB"]:
        if bank.lower() in text.lower():
            return bank
    return "Unknown"


_PARSERS = [
    _try_upi_mandate,       # NEW — multi-line UPI Mandate format
    _try_upi_generic,       # NEW — generic UPI paid/payment messages
    _try_axis_spent,
    _try_hdfc_debited,
    _try_hdfc_alert,
    _try_sbi,
    _try_sbi_upi,
    _try_icici,
    _try_kotak,
    _try_generic_inr_debited,
]

# Debit signal — broad enough to catch "Sent Rs.", "mandate", "UPI", etc.
_DEBIT_KEYWORDS = re.compile(
    r"\b(debited|spent|debit|withdrawn|purchase|payment|paid|charged|sent\s+rs|mandate|transferred|auto.?debit)\b",
    re.IGNORECASE
)
_CREDIT_ONLY_KEYWORDS = re.compile(
    r"\b(credited|received|deposit|refund|cashback|OTP|password)\b", re.IGNORECASE
)


def is_debit_sms(text: str) -> bool:
    has_debit = bool(_DEBIT_KEYWORDS.search(text))
    # Only reject if it's PURELY a credit message with no debit signal
    has_credit_only = bool(_CREDIT_ONLY_KEYWORDS.search(text)) and not has_debit
    return has_debit and not has_credit_only


def parse_sms(text: str, api_key: str = "", use_ai: bool = True) -> Optional[dict]:
    """
    1. Check if it looks like a debit message.
    2. Try all regex parsers in order.
    3. Fall back to AI for any format that slips through.
    """
    if not is_debit_sms(text):
        return None

    for parser in _PARSERS:
        result = parser(text)
        if result:
            result["raw_text"] = text
            return result

    # AI fallback — catches any format regex doesn't handle
    if use_ai and api_key:
        return _parse_sms_with_ai(text, api_key)

    return None


def _parse_sms_with_ai(text: str, api_key: str) -> Optional[dict]:
    """Use the LLM to extract fields from any unrecognised SMS format."""
    from ai_service import _chat
    import json as _json

    prompt = f"""Extract debit transaction details from this Indian bank/UPI SMS.
Return ONLY valid JSON — no markdown, no explanation:
{{"amount": <rupee amount as number>, "date": "YYYY-MM-DD", "merchant": "<business/payee name>", "bank": "<sending bank name>"}}

Date rules: 19/04/26 → 2026-04-19, 20-May-26 → 2026-05-20
Merchant: the business being paid (NOT the bank). E.g. "Spotify India Pvt Ltd" → "Spotify India Pvt Ltd"

SMS:
{text}"""
    try:
        raw = _chat(api_key, [{"role": "user", "content": prompt}], max_tokens=100)
        data = _json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())
        merchant = str(data.get("merchant", "Unknown"))
        return {
            "amount": float(data["amount"]),
            "date": str(data.get("date", datetime.today().strftime("%Y-%m-%d"))),
            "merchant": merchant,
            "description": merchant,
            "bank": str(data.get("bank", "Unknown")),
            "raw_text": text,
            "parsed_by": "ai",
        }
    except Exception:
        return None


def parse_multiple_sms(bulk_text: str, api_key: str = "") -> list[dict]:
    """Split a block of pasted SMS messages and parse each one."""
    # Split on double newlines first
    messages = re.split(r"\n{2,}", bulk_text.strip())

    # If no blank lines, try splitting on sentence-end + capital letter
    if len(messages) == 1:
        messages = re.split(r"(?<=[.!?])\s+(?=[A-Z])", bulk_text)

    results = []
    for msg in messages:
        msg = msg.strip()
        if len(msg) < 20:
            continue
        parsed = parse_sms(msg, api_key=api_key, use_ai=bool(api_key))
        if parsed:
            results.append(parsed)

    return results
