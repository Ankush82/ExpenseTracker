"""OpenRouter AI calls: categorisation, image extraction, financial advice."""

import base64
import json
import re
import urllib.parse
import urllib.request
from typing import Optional  # noqa: F401

from openai import OpenAI

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Free text models — tried in order on rate limit
FREE_TEXT_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-v4-flash:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "openai/gpt-oss-120b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

FREE_VISION_MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

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

# ---------------------------------------------------------------------------
# Rule-based merchant → category lookup (runs before AI, zero API calls)
# Keys are lowercase substrings matched against merchant + description text.
# ---------------------------------------------------------------------------
_RULES: list[tuple[list[str], str]] = [
    # Food & Dining
    (["swiggy", "bundl techn", "bundltech", "zomato", "mcdonalds", "mcd", "kfc",
      "dominos", "pizza hut", "subway", "burger king", "starbucks", "cafe",
      "restaurant", "biryani", "dhaba", "canteen", "foodpanda", "eatsure",
      "dunzo food", "rebel foods", "faasos", "box8",
      "coca-cola", "pepsi", "coke", "juice", "beverage", "drink",
      "snack", "biscuit", "chips", "maggi", "noodle", "bread", "milk",
      "chocolate", "candy", "ice cream", "dairy"], "Food & Dining"),

    # Groceries
    (["blinkit", "blnkt", "zepto", "bigbasket", "big basket", "grofers",
      "jiomart", "dmart", "reliance fresh", "more supermarket", "nature basket",
      "spencer", "easyday", "spar", "hypercity", "lulu"], "Groceries"),

    # Travel & Transport
    (["ola cabs", "olacabs", "olaelectric", "uber", "rapido", "irctc", "indian railway", "air india", "indigo",
      "spicejet", "vistara", "goair", "akasa", "makemytrip", "mmt", "goibibo",
      "redbus", "cleartrip", "yatra", "metro", "bmtc", "apsrtc", "gsrtc",
      "petrol", "diesel", "fuel", "hp petrol", "iocl", "bpcl", "shell",
      "fasttag", "toll"], "Travel & Transport"),

    # Shopping
    (["amazon", "flipkart", "myntra", "ajio", "nykaa", "meesho", "snapdeal",
      "shopsy", "tatacliq", "reliancedigital", "croma", "vijay sales",
      "trends", "westside", "zara", "h&m", "pantaloons", "lifestyle",
      "max fashion", "shoppers stop"], "Shopping"),

    # Entertainment
    (["netflix", "nflx", "spotify", "prime video", "hotstar", "disney",
      "zee5", "sonyliv", "bookmyshow", "pvr", "inox", "cinepolis",
      "youtube premium", "apple music", "gaana", "jiosaavn", "wynk",
      "loot", "gaming", "playstation", "xbox", "steam"], "Entertainment"),

    # Bills & EMI
    (["emi", "ach d", "nach", "loan", "equated", "hdfc bk loan", "lic ",
      "insurance", "bajaj finserv", "home loan", "car loan", "credit card",
      "nach debit", "mandate", "auto debit", "ecs"], "Bills & EMI"),

    # Utilities
    (["electricity", "bescom", "msedcl", "tata power", "adani electricity",
      "tneb", "bses", "cesc", "water", "gas", "piped gas", "indane",
      "hp gas", "bharat gas", "airtel", "jio", "bsnl", "vodafone", "vi ",
      "broadband", "internet", "postpaid", "prepaid recharge", "tata sky",
      "dish tv", "d2h", "sun direct"], "Utilities"),

    # Healthcare
    (["apollo", "fortis", "medplus", "practo", "1mg", "pharmeasy", "netmeds",
      "hospital", "clinic", "pharmacy", "chemist", "medicine", "diagnostic",
      "thyrocare", "dr lal", "healthians", "dentist", "optician",
      "health insurance", "max hospital", "aiims"], "Healthcare"),

    # Sports & Fitness
    (["decathlon", "cult.fit", "cultfit", "gymshark", "gym", "fitness",
      "sports", "swim", "cricket", "football", "badminton", "tennis",
      "yoga", "zumba", "crossfit", "gold gym", "anytime fitness",
      "nike", "adidas", "puma", "reebok", "asics", "new balance",
      "swimming cap", "dumbbell", "protein", "whey"], "Sports & Fitness"),

    # Personal Care
    (["salon", "haircut", "parlour", "parlor", "nykaa fashion", "mamaearth",
      "wow skin", "plum", "minimalist", "sugar cosmetics", "lakme",
      "lotus", "biotique", "himalaya", "urban company", "urbanclap",
      "grooming", "spa", "massage"], "Personal Care"),

    # Indulgence
    (["bar ", "pub ", "alcohol", "beer", "wine", "whisky", "liquor",
      "luxury", "jewellery", "jewelry", "tanishq", "malabar", "kalyan",
      "gold", "diamond", "watch", "rolex", "coach", "gucci", "louis vuitton",
      "armani", "versace"], "Indulgence"),
]


def rule_based_category(merchant: str, description: str = "") -> Optional[str]:
    """Return a category if any keyword rule matches, else None."""
    text = " " + (merchant + " " + description).lower() + " "
    for keywords, category in _RULES:
        for kw in keywords:
            # Short keywords (<=4 chars) need word boundaries to avoid false hits
            if len(kw) <= 4:
                if re.search(r"\b" + re.escape(kw.strip()) + r"\b", text):
                    return category
            elif kw in text:
                return category
    return None


def _client(api_key: str) -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE, api_key=api_key)


def _chat(api_key: str, messages: list, max_tokens: int = 20) -> str:
    """Call free text models in sequence until one responds."""
    last_err = "No text models available"
    for model in FREE_TEXT_MODELS:
        try:
            resp = _client(api_key).chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = str(e)
            continue
    raise RuntimeError(last_err)


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError("No JSON found in response")


def _match_category(raw: str) -> str:
    """Fuzzy-match a model's reply to a valid category."""
    raw = raw.strip().strip('"').strip("'")
    if raw in CATEGORIES:
        return raw
    for cat in CATEGORIES:
        if cat.lower() in raw.lower() or raw.lower() in cat.lower():
            return cat
    return "Other"


# ---------------------------------------------------------------------------
# Merchant name lookup (cryptic UPI codes → real business names)
# ---------------------------------------------------------------------------

def _ddg_search(query: str) -> str:
    """DuckDuckGo instant-answer API — free, no key needed."""
    url = "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1".format(
        urllib.parse.quote(query)
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read())
        return (
            data.get("Abstract")
            or (data.get("RelatedTopics") or [{}])[0].get("Text", "")
        )
    except Exception:
        return ""


def lookup_merchant(api_key: str, merchant_code: str) -> str:
    """
    Resolve a cryptic merchant code to a real business name.
    E.g. 'HEISETASSE BEVE' → 'Thirdwave Coffee'
    1. DuckDuckGo instant answer
    2. AI reasoning fallback
    Returns the resolved name, or the original if nothing found.
    """
    # Skip lookup if it already looks like a real name
    words = merchant_code.split()
    if len(words) >= 2 and not merchant_code.isupper():
        return merchant_code

    # 1. DuckDuckGo
    snippet = _ddg_search(f"{merchant_code} India merchant payment UPI")
    if snippet and len(snippet) > 20:
        # Ask AI to extract just the business name from the snippet
        prompt = f"""Given this search result snippet about a payment merchant, extract ONLY the real business/brand name (1-4 words). If unclear, return the original code.

Merchant code: {merchant_code}
Search snippet: {snippet}

Reply with ONLY the business name, nothing else."""
        try:
            name = _chat(api_key, [{"role": "user", "content": prompt}], max_tokens=15).strip()
            if name and name.lower() not in ("unknown", "none", merchant_code.lower()):
                return name
        except Exception:
            pass

    # 2. AI reasoning alone
    prompt = f"""This is a cryptic merchant code from an Indian bank/UPI payment SMS: "{merchant_code}"

Common patterns: BUNDL TECHN = Swiggy, HEISETASSE BEVE = Thirdwave Coffee (German: hot cup + beverage).

What real Indian business does "{merchant_code}" most likely refer to?
Reply with ONLY the business name (2-5 words). If genuinely unknown, reply "Unknown"."""
    try:
        name = _chat(api_key, [{"role": "user", "content": prompt}], max_tokens=15).strip()
        if name and name.lower() != "unknown":
            return name
    except Exception:
        pass

    return merchant_code


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
    # 1. Rule-based — instant, no API call
    rule = rule_based_category(merchant, description)
    if rule:
        return rule

    # 2. AI fallback with a richer, few-shot prompt
    cat_list = "\n".join(f"- {c}" for c in CATEGORIES)
    prompt = f"""You categorise Indian bank transactions. Pick exactly one category from this list:
{cat_list}

Examples:
- "BUNDL TECHN" → Food & Dining  (Swiggy)
- "ZOMATO" → Food & Dining
- "BLINKIT" → Groceries
- "UBER" → Travel & Transport
- "NETFLIX" → Entertainment
- "ACH D EMI" → Bills & EMI
- "BESCOM" → Utilities
- "APOLLO PHARMACY" → Healthcare
- "DECATHLON" → Sports & Fitness

Transaction:
Merchant: {merchant or description}
Amount: INR {amount:,.2f}
Bank note: {bank}

Reply with ONLY the category name, nothing else."""

    try:
        return _match_category(_chat(api_key, [{"role": "user", "content": prompt}], max_tokens=20))
    except Exception:
        return "Other"


def categorize_item(api_key: str, item_name: str) -> str:
    """Categorise a single receipt line item."""
    # Rule-based first
    rule = rule_based_category(item_name)
    if rule:
        return rule

    cat_list = ", ".join(CATEGORIES)
    prompt = f"""Categorise this item into one of: {cat_list}

Examples:
- "Coca-Cola Diet Coke" → Food & Dining
- "Swimming Cap" → Sports & Fitness
- "Protein powder" → Sports & Fitness
- "Shampoo" → Personal Care
- "Laptop bag" → Shopping
- "Paracetamol" → Healthcare

Item: {item_name}
Reply with ONLY the category name."""

    try:
        return _match_category(_chat(api_key, [{"role": "user", "content": prompt}], max_tokens=20))
    except Exception:
        return "Other"


# ---------------------------------------------------------------------------
# Image / screenshot extraction
# ---------------------------------------------------------------------------

def extract_from_screenshot(api_key: str, image_bytes: bytes, mime: str = "image/jpeg") -> dict:
    """Try each free vision model in turn; return on first success."""
    b64 = base64.b64encode(image_bytes).decode()

    prompt = """You are extracting data from an Indian payment or order screenshot.
Return ONLY a JSON object — no markdown, no explanation:
{
  "merchant": "<app or store name, e.g. Blinkit, Swiggy, Amazon>",
  "total_amount": <final amount paid, as a plain number>,
  "date": "<YYYY-MM-DD if visible, else null>",
  "items": [
    {"name": "<item name>", "quantity": "<e.g. 1x or 180ml x 3>", "price": <number>}
  ]
}
Use the final bill total (after discounts) for total_amount.
List every individual item you can see."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }
    ]

    last_error = "No vision models available"
    for model in FREE_VISION_MODELS:
        try:
            resp = _client(api_key).chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=400,
            )
            result = _extract_json(resp.choices[0].message.content)
            # Auto-categorise each item using rules before returning
            for item in result.get("items", []):
                item["category"] = categorize_item(api_key, item.get("name", ""))
            return result
        except Exception as e:
            last_error = str(e)
            continue

    return {
        "merchant": "Unknown",
        "total_amount": 0.0,
        "date": None,
        "items": [],
        "error": f"All vision models failed. Last: {last_error}",
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

Write a short analysis with these sections:
1. **Overall Assessment** (2-3 sentences)
2. **Top Spending Areas** (biggest categories, are they reasonable?)
3. **3 Actionable Tips** specific to this data
4. **Estimated Monthly Savings** if tips are followed

Be specific, practical, use INR amounts. Under 350 words."""

    try:
        return _chat(api_key, [{"role": "user", "content": prompt}], max_tokens=400)
    except Exception as e:
        return f"Could not generate advice: {e}"
