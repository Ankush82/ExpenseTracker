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
    ICICI Bank Acct XX1234 debited for Rs 500.00 on 20-May-2026; ... trf to MERCHANT
    """
    m = re.search(
        r"ICICI Bank.*?debited for Rs\.?\s*([\d,]+(?:\.\d+)?)\s+on\s+(\d{2}-[A-Za-z]+-\d{2,4}).*?(?:trf to|at)\s+(.+?)(?:;|\.|$)",
        text, re.IGNORECASE
    )
    if m:
        desc = m.group(3).strip()
        return {
            "amount": _parse_amount(m.group(1)),
            "date": _normalize_date(m.group(2)),
            "merchant": desc,
            "description": desc,
            "bank": "ICICI Bank",
        }
    return None


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

    # Pattern: debited Rs/INR X
    m = re.search(
        r"debited\s+(?:Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)",
        text, re.IGNORECASE
    )
    if m:
        # Try to find a date
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


_PARSERS = [
    _try_axis_spent,
    _try_hdfc_debited,
    _try_hdfc_alert,
    _try_sbi,
    _try_sbi_upi,
    _try_icici,
    _try_kotak,
    _try_generic_inr_debited,
]

# Keywords that indicate a debit (not credit/OTP/balance)
_DEBIT_KEYWORDS = re.compile(
    r"\b(debited|spent|debit|withdrawn|purchase|payment|paid|charged)\b", re.IGNORECASE
)
_CREDIT_KEYWORDS = re.compile(
    r"\b(credited|received|deposit|refund|cashback|OTP|password)\b", re.IGNORECASE
)


def is_debit_sms(text: str) -> bool:
    has_debit = bool(_DEBIT_KEYWORDS.search(text))
    has_credit_only = bool(_CREDIT_KEYWORDS.search(text)) and not has_debit
    return has_debit and not has_credit_only


def parse_sms(text: str) -> Optional[dict]:
    """
    Parse a single SMS string. Returns a dict with keys:
      amount, merchant, date, description, bank
    or None if not a recognisable debit SMS.
    """
    if not is_debit_sms(text):
        return None

    for parser in _PARSERS:
        result = parser(text)
        if result:
            result["raw_text"] = text
            return result

    return None


def parse_multiple_sms(bulk_text: str) -> list[dict]:
    """
    Split a block of pasted SMS messages and parse each one.
    Returns list of parsed debit transactions.
    """
    # Split on blank lines or lines starting with common bank names / UPDATE
    messages = re.split(r"\n{2,}", bulk_text.strip())

    # If no blank-line separation, try splitting on sentence patterns
    if len(messages) == 1:
        messages = re.split(r"(?<=[.!?])\s+(?=[A-Z])", bulk_text)

    results = []
    for msg in messages:
        msg = msg.strip()
        if len(msg) < 20:
            continue
        parsed = parse_sms(msg)
        if parsed:
            results.append(parsed)

    return results
