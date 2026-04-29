import streamlit as st
import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from PIL import Image
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from openai import AzureOpenAI

from app.vision import AzureDocumentClient
from app.rebalancer import load_all_funds, optimize_portfolio
from app.personas import PERSONAS, get_targets_for_persona
from app.schemas import PortfolioTargets

# Page Config
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
load_dotenv(override=True)

# Custom CSS to change slider color and font size
st.markdown("""
    <style>
        /* Slider handle - adapt to theme using CSS variables */
        div[role="slider"] {
            background-color: var(--text-color, #FAFAFA) !important;
            border: 2px solid #808080 !important;
            box-shadow: 0 0 5px rgba(0,0,0,0.2) !important;
        }
        
        /* Slider active track color - keep it grey */
        .stSlider [data-baseweb="slider"] > div > div > div {
            background-image: linear-gradient(to right, #808080 0%, #808080 100%) !important;
        }
        
        /* Fix for multi-handle sliders (Global partition) */
        .stSlider [data-baseweb="slider"] > div > div > div > div {
            background-color: #808080 !important;
        }

        /* Increase font size of numbers on the slider (Min/Max and Value) */
        .stSlider [data-testid="stTickBarMin"], 
        .stSlider [data-testid="stTickBarMax"],
        .stSlider [data-baseweb="slider"] div {
            font-size: 1.4rem !important;
            font-weight: 600 !important;
        }
        
        /* Specific target for the floating value label above the handle */
        .stSlider [data-baseweb="slider"] span {
            font-size: 1.6rem !important;
            font-weight: bold !important;
            color: var(--text-color) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Shared Clients
client = AzureDocumentClient()
oai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# Initialize Session State
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "planning_results" not in st.session_state:
    st.session_state.planning_results = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am your AI Financial Agent. I can see all your GM and Truist funds. How can I help you today?"}
    ]

# --- AUTHENTICATION GATE ---
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] == os.getenv("APP_USERNAME", "admin")
            and st.session_state["password"] == os.getenv("APP_PASSWORD", "demo123")
        ):
            st.session_state["authenticated"] = True
            del st.session_state["password"]  # don't store password
            del st.session_state["username"]
        else:
            st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        # Display login form
        st.title("🔒 Portfolio Optimizer Secure Access")
        st.info("This application is private. Please enter your credentials to continue.")
        
        with st.form("login_form"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Login", on_click=password_entered)
        
        if "username" in st.session_state and not st.session_state["authenticated"]:
            st.error("😕 User not known or password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()  # Do not continue if not authenticated

# Sidebar
st.sidebar.title("🤖 Portfolio Agent")
st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"authenticated": False}))
app_mode = st.sidebar.selectbox("Choose Mode", ["Vision Lab", "Portfolio Rebalancer", "AI Agent Chat"])

def get_status_icon(file_path, output_dir):
    json_path = output_dir / f"{file_path.stem}.json"
    return "✅" if json_path.exists() else "⏳"

# --- MODE 1: VISION LAB ---
if app_mode == "Vision Lab":
    st.title("👁️ Vision Lab: Fund Data Extraction")
    st.markdown("Analyze fund prospectuses and extract structured financial data using GPT-4o.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Fund Images")
        input_root = Path("input/funds")
        output_root = Path("outputs/funds")
        all_images = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            all_images.extend(list(input_root.rglob(ext)))
        
        selected_img_path = st.selectbox(
            "Select an image to analyze",
            all_images,
            format_func=lambda x: f"{get_status_icon(x, output_root / x.parent.name)} {x.parent.name}/{x.name}"
        )

        if selected_img_path:
            img = Image.open(selected_img_path)
            st.image(img, caption=selected_img_path.name, use_container_width=True)
            account_source = selected_img_path.parent.name
            output_dir = output_root / account_source
            output_file = output_dir / f"{selected_img_path.stem}.json"

            if st.button("🚀 Extract Data", type="primary"):
                with st.spinner(f"Extracting {selected_img_path.name}..."):
                    try:
                        extracted = client.extract_funds_from_file(selected_img_path, account_source=account_source)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        with open(output_file, "w") as f:
                            json.dump(extracted, f, indent=2)
                        st.success("Extraction Complete!")
                    except Exception as e:
                        st.error(f"Extraction Failed: {e}")

    with col2:
        st.subheader("Extracted Data (JSON)")
        if selected_img_path:
            output_file = output_root / selected_img_path.parent.name / f"{selected_img_path.stem}.json"
            if output_file.exists():
                with open(output_file, "r") as f:
                    data = json.load(f)
                st.json(data)
                if data:
                    df = pd.DataFrame(data)
                    st.table(df[["name", "symbol", "expense_ratio"]])
                    if "sectors" in data[0] and data[0]["sectors"]:
                        st.markdown("**Sector Exposure**")
                        sector_df = pd.DataFrame(list(data[0]["sectors"].items()), columns=["Sector", "Weight"])
                        fig = px.pie(sector_df, values="Weight", names="Sector", hole=0.4)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No extracted data found for this image.")

# --- MODE 2: PORTFOLIO REBALANCER ---
elif app_mode == "Portfolio Rebalancer":
    st.title("🎯 Portfolio Rebalancer: Multi-Agent Optimizer")
    
    # 1. Config Row 1: Persona (Aligned)
    c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
    with c1:
        persona_key = st.selectbox("Agent Persona", list(PERSONAS.keys()), format_func=lambda x: PERSONAS[x].name)
        persona = PERSONAS[persona_key]
    with c2:
        st.info(f"**Philosophy:** {persona.philosophy}")

    # 2. Config Row 2: Balances
    c3, c4 = st.columns(2)
    with c3:
        gm_balance = st.number_input("GM 401k Balance ($)", value=100000.0, step=1000.0, key="gm_bal")
    with c4:
        truist_balance = st.number_input("Truist 401k Balance ($)", value=50000.0, step=1000.0, key="truist_bal")

    st.divider()

    # 3. Allocation State Logic (Sync with Persona Defaults)
    if "last_persona" not in st.session_state or st.session_state.last_persona != persona_key:
        s = persona.default_targets
        st.session_state.last_persona = persona_key
        
        # Directly update the slider keys to force UI update
        dom = int(s.get("domestic_total", 0.6) * 100)
        intl = int(s.get("intl_total", 0.4) * 100)
        st.session_state.g_slider = (dom, dom + intl)
        
        lg = int(s.get("lg_cap_share", 0.8) * 100)
        mid = int(s.get("mid_cap_share", 0.1) * 100)
        st.session_state.d_slider = (lg, lg + mid)
        
        st.session_state.ti_slider = int(s.get("intl_total_share", 0.8) * 100)

    # 4. Global Partition Slider
    st.subheader("📊 Global Allocation")
    # st.write("Drag handles to partition: :blue[**Domestic**] | :orange[**International**] | :green[**Cash/Free**]")
    
    g_split = st.slider(
        "Drag Slider to allocate percent to :blue[**Domestic**] | :orange[**International**] | :green[**Cash/Free**]",
        0, 100, 
        key="g_slider"
    )
    
    dom_pct = g_split[0]
    intl_pct = g_split[1] - g_split[0]
    cash_pct = 100 - g_split[1]
    
    # Proportional Display Row
    col_weights = []
    if dom_pct > 0: col_weights.append(max(dom_pct, 10))
    if intl_pct > 0: col_weights.append(max(intl_pct, 10))
    if cash_pct > 0: col_weights.append(max(cash_pct, 10))
    
    if col_weights:
        cols = st.columns(col_weights)
        curr_idx = 0
        if dom_pct > 0:
            cols[curr_idx].markdown(f"<div style='background-color: #1f77b4; height: 10px; border-radius: 5px; margin-bottom: 5px;'></div>", unsafe_allow_html=True)
            cols[curr_idx].markdown(f"<h3 style='color: #1f77b4;'>US: {dom_pct}%</h3>", unsafe_allow_html=True)
            curr_idx += 1
        if intl_pct > 0:
            cols[curr_idx].markdown(f"<div style='background-color: #ff7f0e; height: 10px; border-radius: 5px; margin-bottom: 5px;'></div>", unsafe_allow_html=True)
            cols[curr_idx].markdown(f"<h3 style='color: #ff7f0e;'>Intl: {intl_pct}%</h3>", unsafe_allow_html=True)
            curr_idx += 1
        if cash_pct > 0:
            cols[curr_idx].markdown(f"<div style='background-color: #2ca02c; height: 10px; border-radius: 5px; margin-bottom: 5px;'></div>", unsafe_allow_html=True)
            cols[curr_idx].markdown(f"<h3 style='color: #2ca02c;'>Cash/Free: {cash_pct}%</h3>", unsafe_allow_html=True)
            curr_idx += 1

    st.divider()

    # 5. Asset Class Breakdowns (Partition Sliders)
    st.subheader("🧩 Asset Class Breakdowns")
    
    col_sub_left, col_sub_right = st.columns(2)
    
    with col_sub_left:
        st.markdown("#### :blue[Domestic Breakdown]")
        d_split = st.slider(
            "Drag Slider to allocate percentage to **Large Cap** | **Mid Cap** | **Small Cap**:",
            0, 100, 
            key="d_slider"
        )
        lg_s = d_split[0]
        mid_s = d_split[1] - d_split[0]
        sm_s = 100 - d_split[1]
        
        st.markdown(f"<div style='display: flex; height: 10px; border-radius: 5px; margin-top: 10px; margin-bottom: 5px;'>"
                    f"<div style='background-color: #1f77b4; width: {lg_s}%;'></div>"
                    f"<div style='background-color: #4da6ff; width: {mid_s}%;'></div>"
                    f"<div style='background-color: #99ccff; width: {sm_s}%;'></div>"
                    f"</div>", unsafe_allow_html=True)
        st.markdown(f"**Large:** {lg_s}% | **Mid:** {mid_s}% | **Small:** {sm_s}%")

    with col_sub_right:
        st.markdown("#### :orange[International Breakdown]")
        ti_s = st.slider(
            "Drag Slider to allocate percentage to **Total International* | **Emerging Markets**",
            0, 100, 
            key="ti_slider"
        )
        em_s = 100 - ti_s
        st.markdown(f"<div style='display: flex; height: 10px; border-radius: 5px; margin-top: 10px; margin-bottom: 5px;'>"
                    f"<div style='background-color: #ff7f0e; width: {ti_s}%;'></div>"
                    f"<div style='background-color: #ffb366; width: {em_s}%;'></div>"
                    f"</div>", unsafe_allow_html=True)
        st.markdown(f"**Total Intl:** {ti_s}% | **Emerging:** {em_s}%")

    # 6. Rebalance Execution
    targets = PortfolioTargets(
        domestic_total=dom_pct / 100.0,
        intl_total=intl_pct / 100.0,
        lg_cap_share=lg_s / 100.0,
        mid_cap_share=mid_s / 100.0,
        sm_cap_share=sm_s / 100.0,
        intl_total_share=ti_s / 100.0,
        emerging_markets_share=em_s / 100.0
    )

    if st.button("🚀 Rebalance Portfolio", type="primary", use_container_width=True):
        with st.spinner("Optimizing..."):
            account_balances = [{"account_name": "GM", "balance": gm_balance}, {"account_name": "Truist", "balance": truist_balance}]
            optimize_portfolio(targets, account_balances, persona_key=persona_key, persona_name=persona.name)
            plan_file = Path("outputs/rebalancer") / persona_key / "plan.json"
            if plan_file.exists():
                with open(plan_file, "r") as f:
                    st.session_state.planning_results = json.load(f)
                summary_file = Path("outputs/rebalancer") / persona_key / "summary.txt"
                if summary_file.exists():
                    with open(summary_file, "r") as f:
                        st.session_state.planning_summary = f.read()

    # Results Section
    st.divider()
    if st.session_state.planning_results:
        res = st.session_state.planning_results
        st.subheader(f"Results: {res['summary']['persona']}")
        total_val = res['summary']['total_value']
        er = res['summary']['aggregate_expense_ratio']
        m1, m2 = st.columns(2)
        m1.metric("Total Portfolio Value", f"${total_val:,.2f}")
        m2.metric("Aggregate Expense Ratio", f"{er:.4f}%")
        
        funds_df = pd.DataFrame(res['selected_funds'])
        for account in funds_df["account_source"].unique():
            st.markdown(f"### 🏦 {account.upper()} Account")
            acc_df = funds_df[funds_df["account_source"] == account]
            display_df = acc_df[["name", "allocated_dollars", "shares", "expense_ratio"]]
            display_df.columns = ["Fund Name", "Dollars", "Shares", "ER (%)"]
            st.table(display_df.style.format({"Dollars": "${:,.2f}", "Shares": "{:,.4f}", "ER (%)": "{:.4f}%"}))
            st.markdown(f"**Account Total:** ${acc_df['allocated_dollars'].sum():,.2f}")
        
        st.markdown("### 📊 Aggregate Sector Exposure")
        sectors = res['summary']['sector_makeup']
        sector_df = pd.DataFrame(list(sectors.items()), columns=["Sector", "Weight"]).sort_values(by="Weight", ascending=False)
        fig = px.bar(sector_df, x="Weight", y="Sector", orientation='h', title="Weighted Sector Exposure")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🧠 AI Executive Advisory")
        if "planning_summary" in st.session_state:
            summary = st.session_state.planning_summary
            if "AI ADVISORY:" in summary:
                advisory = summary.split("AI ADVISORY:")[1].strip()
                st.info(advisory)
            elif "AI ADVISOR:" in summary:
                advisory = summary.split("AI ADVISOR:")[1].strip()
                st.info(advisory)
            else:
                sections = summary.split("-" * 50)
                st.info(sections[-1].strip() if len(sections) > 1 else summary)

# --- MODE 3: AI AGENT CHAT ---
elif app_mode == "AI Agent Chat":
    st.title("🧠 AI Financial Agent")
    st.markdown("Chat with the agent to audit, analyze, and rebalance your portfolio using natural language.")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a financial question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # 1. Intent Gate
            with st.status("🛡️ Checking House Policy Compliance...", expanded=False):
                policy_prompt = f"Classify intent: If finance/401k/investment/prospectus response ALLOWED, else BLOCKED. QUERY: {prompt}"
                try:
                    check = oai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": policy_prompt}], temperature=0)
                    policy_result = check.choices[0].message.content
                except Exception as e:
                    policy_result = "ALLOWED (Filter Override)" if "content_filter" in str(e) else f"BLOCKED: API Error"
                
                if "BLOCKED" in policy_result:
                    full_response = "⚠️ **Compliance Notice:** I am only authorized to discuss financial data. Your query was filtered."
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.stop()
                else:
                    st.write("✅ Query compliance verified.")

            # 2. Agent Loop
            tools = [
                {"type": "function", "function": {"name": "query_prospectus_semantics", "description": "Search prospectuses for risks/fees.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
                {"type": "function", "function": {"name": "list_available_funds", "description": "List all extracted plan funds."}},
                {"type": "function", "function": {"name": "check_compliance_guardrails", "description": "Validate rebalance plan.", "parameters": {"type": "object", "properties": {"plan_summary_json": {"type": "string"}}, "required": ["plan_summary_json"]}}},
                {"type": "function", "function": {"name": "fetch_live_market_data", "description": "Fetch real-time ticker data.", "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}},
                {"type": "function", "function": {"name": "rebalance_portfolio", "description": "Math rebalance. Needs balances.", "parameters": {"type": "object", "properties": {"domestic_total": {"type": "number"}, "intl_total": {"type": "number"}, "gm_balance": {"type": "number"}, "truist_balance": {"type": "number"}}, "required": ["domestic_total", "intl_total", "gm_balance", "truist_balance"]}}}
            ]

            msgs = [{"role": "system", "content": "You are a professional Wealth Management Agent. Use tools for accuracy. Ask for GM/Truist balances before rebalancing."}]
            for m in st.session_state.messages: msgs.append({"role": m["role"], "content": m["content"]})

            for _ in range(5):
                response = oai_client.chat.completions.create(model="gpt-4o", messages=msgs, tools=tools, tool_choice="auto")
                response_message = response.choices[0].message
                msgs.append(response_message)
                if response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        fname = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)
                        with st.status(f"🛠️ Tool: `{fname}`", expanded=False):
                            if fname == "list_available_funds":
                                funds = load_all_funds()
                                tool_result = "\n".join([f"- {f.name}: {f.expense_ratio}% [{f.account_source.upper()}]" for f in funds])
                            elif fname == "query_prospectus_semantics":
                                from app.rag import query_prospectus_semantics as run_query
                                tool_result = run_query(args.get("query"))
                            elif fname == "fetch_live_market_data":
                                import yfinance as yf
                                t = yf.Ticker(args.get("ticker", "VTI"))
                                tool_result = f"Price ${t.info.get('regularMarketPrice', 'N/A')}, Name: {t.info.get('longName')}"
                            elif fname == "check_compliance_guardrails":
                                d = json.loads(args["plan_summary_json"])
                                tool_result = "✅ PASSED" if (d.get("aggregate_expense_ratio", 0) < 0.5 and d.get("targets", {}).get("intl_total", 0) > 0.1) else "❌ FAILED"
                            elif fname == "rebalance_portfolio":
                                t = PortfolioTargets(domestic_total=args["domestic_total"], intl_total=args["intl_total"])
                                b = [{"account_name": "GM", "balance": args["gm_balance"]}, {"account_name": "Truist", "balance": args["truist_balance"]}]
                                res = optimize_portfolio(t, b)
                                tool_result = json.dumps({"summary": res["summary"], "fund_selections": res["selected_funds"]}, indent=2)
                        msgs.append({"tool_call_id": tool_call.id, "role": "tool", "name": fname, "content": tool_result})
                else:
                    full_response = response_message.content
                    break
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
