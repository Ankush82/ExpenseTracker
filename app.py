"""Expense Tracker — Streamlit App."""

import io
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ai_service
import database as db
import sms_parser

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Expense Tracker",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
db.init_db()

if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.display_name = None

if "openrouter_key" not in st.session_state:
    # Load from Streamlit secrets if available
    try:
        st.session_state.openrouter_key = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        st.session_state.openrouter_key = ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "Food & Dining": "#FF6B6B",
    "Groceries": "#4ECDC4",
    "Sports & Fitness": "#45B7D1",
    "Travel & Transport": "#96CEB4",
    "Shopping": "#FFEAA7",
    "Entertainment": "#DDA0DD",
    "Bills & EMI": "#F0A500",
    "Healthcare": "#6BCB77",
    "Utilities": "#A8DADC",
    "Personal Care": "#FFB3BA",
    "Indulgence": "#C9B1FF",
    "Other": "#B0B0B0",
}


def format_inr(amount: float) -> str:
    return f"₹{amount:,.2f}"


def api_key() -> str:
    return st.session_state.openrouter_key


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------

def show_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<h1 style='text-align:center;'>💸 Expense Tracker</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center;color:#888;'>Track. Categorise. Save.</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="user1 / user2 / user3")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", use_container_width=True)

        if submitted:
            result = db.verify_user(username.strip(), password.strip())
            if result:
                st.session_state.user_id = result[0]
                st.session_state.display_name = result[1]
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.caption("Default credentials — user1/pass1, user2/pass2, user3/pass3")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def show_sidebar():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.display_name}")

        # Pending badge
        pending_df = db.get_pending_transactions(st.session_state.user_id)
        n_pending = len(pending_df)
        if n_pending:
            st.warning(f"🔔 {n_pending} SMS awaiting review")

        st.divider()

        inbox_label = f"📥 SMS Inbox ({n_pending})" if n_pending else "📥 SMS Inbox"
        page = st.radio(
            "Navigate",
            ["📊 Dashboard", inbox_label, "📱 Add SMS", "🖼️ Upload Screenshot", "🤖 AI Advisor", "⚙️ Settings"],
            label_visibility="collapsed",
        )
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user_id = None
            st.session_state.display_name = None
            st.rerun()

    # Normalise label back so routing works regardless of badge count
    if page.startswith("📥"):
        page = "📥 SMS Inbox"
    return page


# ---------------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------------

def show_dashboard():
    st.title("📊 Dashboard")

    all_users = db.get_all_users()
    user_options = {row["display_name"]: row["id"] for _, row in all_users.iterrows()}

    # Filters
    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_users = st.multiselect(
                "Users",
                options=list(user_options.keys()),
                default=list(user_options.keys()),
            )

        with col2:
            today = date.today()
            start_date = st.date_input("From", value=today.replace(day=1))

        with col3:
            end_date = st.date_input("To", value=today)
            selected_category = st.selectbox(
                "Category", ["All"] + db.CATEGORIES
            )

    selected_user_ids = [user_options[u] for u in selected_users] if selected_users else []

    df = db.get_transactions(
        user_ids=selected_user_ids if selected_user_ids else None,
        start_date=str(start_date),
        end_date=str(end_date),
        category=selected_category,
    )

    # Summary cards
    st.divider()
    c1, c2, c3, c4 = st.columns(4)

    total_spend = df["amount"].sum() if not df.empty else 0
    today_df = df[df["date"] == str(today)] if not df.empty else pd.DataFrame()
    today_spend = today_df["amount"].sum() if not today_df.empty else 0
    tx_count = len(df)
    avg_tx = df["amount"].mean() if not df.empty else 0

    c1.metric("Total Spend", format_inr(total_spend))
    c2.metric("Today's Spend", format_inr(today_spend))
    c3.metric("Transactions", tx_count)
    c4.metric("Avg Transaction", format_inr(avg_tx))

    if df.empty:
        st.info("No transactions found for the selected filters. Add some via **Add SMS** or **Upload Screenshot**.")
        return

    st.divider()

    # Charts row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Spending by Category")
        cat_df = df.groupby("category")["amount"].sum().reset_index()
        fig_pie = px.pie(
            cat_df,
            values="amount",
            names="category",
            color="category",
            color_discrete_map=CATEGORY_COLORS,
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        st.subheader("Daily Spending Trend")
        daily_df = df.groupby("date")["amount"].sum().reset_index()
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        fig_line = px.bar(
            daily_df,
            x="date",
            y="amount",
            labels={"amount": "Amount (INR)", "date": "Date"},
            color_discrete_sequence=["#4ECDC4"],
        )
        fig_line.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_line, use_container_width=True)

    # User comparison (only when multiple users selected)
    if len(selected_users) > 1:
        st.subheader("Spending by User")
        user_df = df.groupby("user")["amount"].sum().reset_index()
        fig_user = px.bar(
            user_df,
            x="user",
            y="amount",
            color="user",
            labels={"amount": "Amount (INR)", "user": ""},
        )
        fig_user.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_user, use_container_width=True)

    # Monthly category heatmap-style table
    st.subheader("Category Breakdown")
    pivot = df.groupby(["date", "category"])["amount"].sum().unstack(fill_value=0)
    st.dataframe(
        pivot.style.format("₹{:,.0f}").background_gradient(cmap="YlOrRd", axis=None),
        use_container_width=True,
    )

    # Transaction table
    st.subheader("Transactions")
    display_df = df[["date", "user", "merchant", "amount", "category", "source", "notes"]].copy()
    display_df["amount"] = display_df["amount"].apply(format_inr)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Delete a transaction
    with st.expander("Delete a transaction"):
        tx_id = st.number_input("Transaction ID to delete", min_value=1, step=1)
        if st.button("Delete", type="primary"):
            db.delete_transaction(int(tx_id))
            st.success(f"Transaction {tx_id} deleted.")
            st.rerun()


# ---------------------------------------------------------------------------
# Add SMS page
# ---------------------------------------------------------------------------

def show_add_sms():
    st.title("📱 Add SMS Transactions")
    st.caption(
        "Paste your bank debit SMS messages below (one or more). "
        "Supported banks: Axis Bank, HDFC, SBI, ICICI, Kotak and most others."
    )
    st.info(
        "**iPhone tip:** Open Messages app → find your bank sender → copy the SMS text → paste here. "
        "You can paste multiple messages separated by a blank line."
    )

    sms_text = st.text_area(
        "Paste SMS messages here",
        height=200,
        placeholder="Spent INR 2095 Axis Bank Card no. XX4830 19-05-26 09:17:50 IST BUNDL TECHN ...\n\nUPDATE: INR 12,200.00 debited from HDFC Bank XX7750 on 20-MAY-26. Info: ACH D- BOI EMI...",
    )

    if st.button("Parse SMS", type="primary", disabled=not sms_text.strip()):
        parsed = sms_parser.parse_multiple_sms(sms_text)
        if not parsed:
            st.error("No debit transactions detected. Make sure you pasted bank debit SMS messages.")
        else:
            st.session_state["parsed_sms"] = parsed
            st.success(f"Found {len(parsed)} debit transaction(s).")

    if "parsed_sms" in st.session_state and st.session_state["parsed_sms"]:
        st.divider()
        st.subheader("Review & Save")

        parsed_list = st.session_state["parsed_sms"]
        to_save = []

        for i, tx in enumerate(parsed_list):
            with st.container(border=True):
                cols = st.columns([2, 1, 2, 2, 1])

                with cols[0]:
                    merchant = st.text_input(
                        "Merchant", value=tx.get("merchant", ""), key=f"merchant_{i}"
                    )
                with cols[1]:
                    amount = st.number_input(
                        "Amount (₹)", value=float(tx.get("amount", 0)), key=f"amount_{i}", min_value=0.0
                    )
                with cols[2]:
                    tx_date = st.date_input(
                        "Date",
                        value=pd.to_datetime(tx.get("date", str(date.today()))).date(),
                        key=f"date_{i}",
                    )
                with cols[3]:
                    # Auto-categorise with AI if key is available
                    default_cat = "Other"
                    if api_key() and st.button(
                        "AI Categorise", key=f"ai_cat_{i}", help="Use AI to suggest a category"
                    ):
                        with st.spinner("Categorising..."):
                            default_cat = ai_service.categorize_transaction(
                                api_key(),
                                merchant,
                                amount,
                                description=tx.get("description", ""),
                                bank=tx.get("bank", ""),
                            )
                        st.session_state[f"cat_val_{i}"] = default_cat

                    category = st.selectbox(
                        "Category",
                        db.CATEGORIES,
                        index=db.CATEGORIES.index(
                            st.session_state.get(f"cat_val_{i}", "Other")
                        ),
                        key=f"category_{i}",
                    )
                with cols[4]:
                    include = st.checkbox("Save", value=True, key=f"include_{i}")

                notes = st.text_input(
                    "Notes (optional)", value="", key=f"notes_{i}", placeholder="e.g. Office lunch"
                )

                if include:
                    to_save.append(
                        {
                            "merchant": merchant,
                            "amount": amount,
                            "date": str(tx_date),
                            "category": category,
                            "notes": notes,
                            "raw_text": tx.get("raw_text", ""),
                        }
                    )

        st.divider()
        if st.button("Save Selected Transactions", type="primary"):
            saved = 0
            for tx in to_save:
                db.save_transaction(
                    user_id=st.session_state.user_id,
                    amount=tx["amount"],
                    merchant=tx["merchant"],
                    date=tx["date"],
                    description=tx["merchant"],
                    category=tx["category"],
                    source="sms",
                    raw_text=tx["raw_text"],
                    notes=tx["notes"],
                )
                saved += 1
            st.success(f"Saved {saved} transaction(s)!")
            del st.session_state["parsed_sms"]
            st.rerun()


# ---------------------------------------------------------------------------
# Upload Screenshot page
# ---------------------------------------------------------------------------

def show_upload_screenshot():
    st.title("🖼️ Upload Payment Screenshot")
    st.caption(
        "Upload a screenshot from Blinkit, Swiggy, Zomato, Amazon, or any payment app. "
        "AI will extract the items and amounts."
    )

    if not api_key():
        st.error("OpenRouter API key not configured. Go to **Settings** to add it.")
        return

    uploaded = st.file_uploader(
        "Choose a screenshot",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )

    if uploaded:
        img_bytes = uploaded.read()
        mime = uploaded.type or "image/jpeg"

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img_bytes, caption="Uploaded screenshot", use_container_width=True)

        with col2:
            if st.button("Extract with AI", type="primary"):
                with st.spinner("Analysing screenshot..."):
                    result = ai_service.extract_from_screenshot(api_key(), img_bytes, mime)

                if "error" in result and not result.get("items"):
                    st.error(f"Extraction failed: {result['error']}")
                else:
                    st.session_state["screenshot_result"] = result
                    st.success("Extraction complete!")

    if "screenshot_result" in st.session_state:
        result = st.session_state["screenshot_result"]
        st.divider()
        st.subheader("Extracted Data — Review & Save")

        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            merchant = col1.text_input("Merchant / App", value=result.get("merchant", ""))
            total_amount = col2.number_input(
                "Total Amount (₹)",
                value=float(result.get("total_amount", 0.0)),
                min_value=0.0,
            )

            raw_date = result.get("date") or str(date.today())
            try:
                parsed_date = pd.to_datetime(raw_date).date()
            except Exception:
                parsed_date = date.today()
            tx_date = col3.date_input("Date", value=parsed_date)

            # Overall category
            if st.button("AI Categorise transaction"):
                with st.spinner("Categorising..."):
                    suggested = ai_service.categorize_transaction(
                        api_key(), merchant, total_amount
                    )
                st.session_state["sc_cat"] = suggested

            overall_category = st.selectbox(
                "Overall Category",
                db.CATEGORIES,
                index=db.CATEGORIES.index(st.session_state.get("sc_cat", "Other")),
            )
            notes = st.text_input("Notes", placeholder="Optional note about this purchase")

        # Line items
        items = result.get("items", [])
        if items:
            st.subheader("Line Items")
            edited_items = []
            for j, item in enumerate(items):
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
                    name = c1.text_input("Item", value=item.get("name", ""), key=f"sc_item_{j}")
                    qty = c2.text_input("Qty", value=str(item.get("quantity", "")), key=f"sc_qty_{j}")
                    price = c3.number_input(
                        "Price (₹)", value=float(item.get("price", 0.0)), min_value=0.0, key=f"sc_price_{j}"
                    )
                    item_cat = c4.selectbox(
                        "Category",
                        db.CATEGORIES,
                        index=db.CATEGORIES.index(
                            ai_service.categorize_item(api_key(), name) if api_key() else "Other"
                        )
                        if not st.session_state.get(f"sc_item_cat_{j}")
                        else db.CATEGORIES.index(st.session_state[f"sc_item_cat_{j}"]),
                        key=f"sc_item_cat_{j}",
                    )
                    edited_items.append({"name": name, "quantity": qty, "price": price, "category": item_cat})

        st.divider()
        if st.button("Save Transaction", type="primary"):
            tx_id = db.save_transaction(
                user_id=st.session_state.user_id,
                amount=total_amount,
                merchant=merchant,
                date=str(tx_date),
                description=merchant,
                category=overall_category,
                source="screenshot",
                raw_text=f"Screenshot upload: {merchant}",
                notes=notes,
            )
            if items:
                db.save_transaction_items(tx_id, edited_items)
            st.success(f"Transaction saved! (ID: {tx_id})")
            del st.session_state["screenshot_result"]
            st.session_state.pop("sc_cat", None)
            st.rerun()


# ---------------------------------------------------------------------------
# AI Advisor page
# ---------------------------------------------------------------------------

def show_ai_advisor():
    st.title("🤖 AI Financial Advisor")
    st.caption("Get personalised money-saving advice based on your actual spending.")

    if not api_key():
        st.error("OpenRouter API key not configured. Go to **Settings** to add it.")
        return

    all_users = db.get_all_users()
    user_options = {row["display_name"]: row["id"] for _, row in all_users.iterrows()}

    col1, col2, col3 = st.columns(3)
    with col1:
        target_users = st.multiselect(
            "Analyse spending for",
            options=list(user_options.keys()),
            default=[st.session_state.display_name],
        )
    with col2:
        period = st.selectbox("Period", ["This Month", "Last 30 Days", "Last 90 Days", "All Time"])
    with col3:
        st.write("")
        st.write("")
        generate = st.button("Generate Advice", type="primary", use_container_width=True)

    today = date.today()
    period_map = {
        "This Month": (today.replace(day=1), today),
        "Last 30 Days": (today - timedelta(days=30), today),
        "Last 90 Days": (today - timedelta(days=90), today),
        "All Time": (None, None),
    }
    start, end = period_map[period]

    selected_ids = [user_options[u] for u in target_users] if target_users else None

    if generate:
        with st.spinner("Analysing spending and generating advice..."):
            summary = db.get_spending_summary_text(
                user_ids=selected_ids,
                start_date=str(start) if start else None,
                end_date=str(end) if end else None,
            )
            period_label = f"{period.lower()} ({start} to {end})" if start else "all time"
            user_label = " & ".join(target_users) if target_users else "All Users"
            advice = ai_service.get_financial_advice(
                api_key(), summary, user_label, period_label
            )

        st.divider()
        st.subheader("Spending Summary")
        with st.expander("View raw summary used for analysis"):
            st.text(summary)

        st.subheader("💡 AI Advice")
        st.markdown(advice)


# ---------------------------------------------------------------------------
# SMS Inbox — pending transactions sent by iPhone Shortcut
# ---------------------------------------------------------------------------

def show_sms_inbox():
    st.title("📥 SMS Inbox")
    st.caption(
        "Transactions captured automatically from your iPhone via the Shortcuts automation "
        "appear here for review before being added to your dashboard."
    )

    pending = db.get_pending_transactions(st.session_state.user_id)

    if pending.empty:
        st.success("No pending transactions — you're all caught up!")
        st.markdown("---")
        st.markdown(
            "**Haven't set up the iPhone Shortcut yet?** Go to ⚙️ Settings → iPhone Shortcut Setup."
        )
        return

    st.info(f"{len(pending)} transaction(s) waiting for your review.")

    for _, row in pending.iterrows():
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 1, 1])
            c1.metric("Amount", f"₹{row['amount']:,.2f}")
            c2.metric("Merchant", row["merchant"] or "Unknown")
            c3.metric("Date", row["date"])

            with st.expander("Original SMS"):
                st.text(row["raw_text"])

            col_cat, col_notes, col_btns = st.columns([2, 2, 1])

            with col_cat:
                # AI-suggest category if key is available
                default_idx = 0
                suggested_key = f"inbox_cat_{row['id']}"
                if suggested_key not in st.session_state and api_key():
                    with st.spinner("AI categorising..."):
                        suggested = ai_service.categorize_transaction(
                            api_key(), row["merchant"], row["amount"]
                        )
                    st.session_state[suggested_key] = suggested

                saved_cat = st.session_state.get(suggested_key, "Other")
                if saved_cat in db.CATEGORIES:
                    default_idx = db.CATEGORIES.index(saved_cat)

                category = st.selectbox(
                    "Category",
                    db.CATEGORIES,
                    index=default_idx,
                    key=f"inbox_sel_{row['id']}",
                )

            with col_notes:
                notes = st.text_input(
                    "Notes",
                    placeholder="e.g. Dinner with team",
                    key=f"inbox_notes_{row['id']}",
                )

            with col_btns:
                st.write("")
                st.write("")
                col_confirm, col_discard = st.columns(2)
                if col_confirm.button("✓ Keep", key=f"confirm_{row['id']}", type="primary"):
                    db.confirm_pending_transaction(int(row["id"]), category, notes)
                    st.toast(f"Saved ₹{row['amount']:,.0f} at {row['merchant']}")
                    st.rerun()
                if col_discard.button("✗ Discard", key=f"discard_{row['id']}"):
                    db.delete_transaction(int(row["id"]))
                    st.toast("Transaction discarded.")
                    st.rerun()

    st.divider()
    if st.button("Confirm All with AI Categories", type="secondary"):
        for _, row in pending.iterrows():
            cat = st.session_state.get(f"inbox_cat_{row['id']}", "Other")
            db.confirm_pending_transaction(int(row["id"]), cat)
        st.success("All transactions confirmed!")
        st.rerun()


# ---------------------------------------------------------------------------
# Settings page
# ---------------------------------------------------------------------------

def show_settings():
    st.title("⚙️ Settings")

    st.subheader("OpenRouter API Key")
    st.caption(
        "Get your API key from [openrouter.ai](https://openrouter.ai). "
        "The key is stored only in your session (not persisted to disk)."
    )

    key_input = st.text_input(
        "API Key",
        value=st.session_state.openrouter_key,
        type="password",
        placeholder="sk-or-...",
    )
    if st.button("Save Key"):
        st.session_state.openrouter_key = key_input.strip()
        st.success("API key saved for this session.")

    st.divider()
    st.subheader("Persistent Key (recommended for deployment)")
    st.markdown("""
Create a `.streamlit/secrets.toml` file in the project root:

```toml
OPENROUTER_API_KEY = "sk-or-your-key-here"
```

Then redeploy — the key will be auto-loaded on startup.
""")

    st.divider()
    st.subheader("📱 iPhone Shortcut Setup")
    st.caption("Set up the automation that silently sends your bank SMS to this app.")

    webhook_token = db.get_config("webhook_token") or "not-set"
    col_tok, col_regen = st.columns([3, 1])
    col_tok.text_input("Your Webhook Token (keep this secret)", value=webhook_token, disabled=True)
    if col_regen.button("Regenerate"):
        import secrets as _sec
        db.set_config("webhook_token", _sec.token_urlsafe(24))
        st.rerun()

    st.markdown("""
**Step-by-step: Create the iPhone Shortcut**

1. Open the **Shortcuts** app on your iPhone.
2. Tap **Automation** (bottom tab) → **+** → **Personal Automation**.
3. Choose **Message Received**.
4. Tap **From** → add your bank sender names
   *(e.g. `AXISBANK`, `HDFCBK`, `SBI`, `ICICIB`, `KOTAKBK`)*
5. Leave "Any Message" selected — our parser filters non-debit SMS server-side.
6. Tap **Next** → **Add Action**.
7. Search for **"Get Details of Messages"** → select it → set detail to **"Message Content"**.
8. Add another action: search **"URL"** → paste your server URL:

```
http://<YOUR-SERVER-IP>:8000/sms
```

9. Add another action: **"Get Contents of URL"** with these settings:
   - Method: **POST**
   - Request Body: **JSON**
   - Add key `message` → value: **Message Content** (the variable from step 7)
   - Add key `user_id` → value: **your user number** (1, 2, or 3)
   - Add header `X-Token` → value: paste your token above

10. Tap **Next** → turn off **"Ask Before Running"** → **Done**.

That's it! Every time a debit SMS arrives, your iPhone silently POSTs it here.
New transactions appear in **📥 SMS Inbox** for your review.
""")

    st.info(
        "**Public access needed:** The iPhone must be able to reach your server over the internet "
        "(not just local WiFi). For home/local use, set up port forwarding on your router "
        "or use a free tunnel like [ngrok](https://ngrok.com): `ngrok http 8000`.\n\n"
        "**On Streamlit Cloud:** The webhook server (port 8000) cannot run there. "
        "Run `python3 run.py` on your own machine or a VPS, then point the Shortcut at that IP."
    )

    st.divider()
    st.subheader("Change Password")
    st.caption("Password changes require a restart currently — contact the admin to reset.")

    st.divider()
    st.subheader("Export Data")
    all_users = db.get_all_users()
    user_options = {row["display_name"]: row["id"] for _, row in all_users.iterrows()}
    export_users = st.multiselect("Export data for", list(user_options.keys()), default=list(user_options.keys()))
    if st.button("Download CSV"):
        ids = [user_options[u] for u in export_users]
        df = db.get_transactions(user_ids=ids)
        csv = df.to_csv(index=False)
        st.download_button(
            "Download expense_data.csv",
            data=csv,
            file_name="expense_data.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not st.session_state.user_id:
        show_login()
        return

    page = show_sidebar()

    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "📥 SMS Inbox":
        show_sms_inbox()
    elif page == "📱 Add SMS":
        show_add_sms()
    elif page == "🖼️ Upload Screenshot":
        show_upload_screenshot()
    elif page == "🤖 AI Advisor":
        show_ai_advisor()
    elif page == "⚙️ Settings":
        show_settings()


if __name__ == "__main__":
    main()
