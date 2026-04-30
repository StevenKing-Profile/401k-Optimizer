from fastmcp import FastMCP
from app.schemas import Fund, PortfolioTargets, AccountBalance
from typing import List, Optional
from pathlib import Path
import json
import os
from dotenv import load_dotenv

# Import our core logic
from app.rebalancer import load_all_funds, optimize_portfolio, get_agent_advisory
from app.vision import AzureDocumentClient

load_dotenv()

# Create the MCP instance
mcp = FastMCP("PortfolioRebalancer")

# --- MCP Tools ---

@mcp.tool()
def ingest_prospectus_pdf(file_path: str, fund_name: str) -> str:
    """
    Ingests a full PDF prospectus into the vector database for semantic searching.
    - file_path: Local path to the PDF file.
    - fund_name: The name of the fund (e.g., 'Truist International').
    """
    from app.rag import ProspectusRAG
    try:
        rag = ProspectusRAG()
        chunks = rag.ingest_pdf(file_path, fund_name)
        return f"Successfully ingested {fund_name}. Document split into {chunks} searchable chunks."
    except Exception as e:
        return f"Ingestion failed: {e}"

@mcp.tool()
def query_prospectus_semantics(query: str, fund_name: Optional[str] = None) -> str:
    """
    Performs a semantic search across ingested prospectuses to answer complex qualitative questions.
    - query: The question (e.g., 'What is the redemption fee?')
    - fund_name: (Optional) Limit search to a specific fund.
    """
    from app.rag import query_prospectus_semantics as run_query
    try:
        return run_query(query, fund_name)
    except Exception as e:
        return f"Semantic query failed: {e}"

@mcp.tool()
def list_available_funds() -> str:
    """
    Returns a summary of all funds currently extracted and available in the system.
    Use this to see what options the rebalancer has to work with.
    """
    funds = load_all_funds()
    if not funds:
        return "No funds found. Please run 'analyze_prospectus' first."
    
    summary = []
    for f in funds:
        summary.append(f"- {f.name} ({f.asset_class.parent} {f.asset_class.sub_class}) | ER: {f.expense_ratio}% | Source: {f.account_source}")
    
    return "\n".join(summary)

@mcp.tool()
def analyze_prospectus(file_path: str, account_source: str) -> str:
    """
    Extracts structured financial data from a fund prospectus image or PDF.
    - file_path: Local path to the image/PDF
    - account_source: 'gm' or 'truist'
    """
    client = AzureDocumentClient()
    path = Path(file_path)
    if not path.exists():
        return f"Error: File {file_path} not found."
    
    try:
        extracted = client.extract_funds_from_file(path, account_source=account_source)
        # Save to the expected output directory
        output_dir = Path("outputs/funds") / account_source.lower()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{path.stem}.json"
        
        with open(output_file, "w") as f:
            json.dump(extracted, f, indent=2)
            
        return f"Successfully extracted {len(extracted)} funds from {path.name}. Saved to {output_file}"
    except Exception as e:
        return f"Extraction failed: {e}"

@mcp.tool()
def check_compliance_guardrails(plan_summary_json: str) -> str:
    """
    Checks a proposed rebalance plan against internal safety and compliance rules.
    Rules: Fee Cap (<0.50%) and Intl Diversification (>10%).
    """
    try:
        data = json.loads(plan_summary_json)
        er = data.get("aggregate_expense_ratio", 0)
        intl = data.get("targets", {}).get("intl_total", 0)
        
        violations = []
        if er > 0.50:
            violations.append(f"FEE WARNING: Portfolio aggregate fee ({er:.2f}%) exceeds the 0.50% house limit.")
        if intl < 0.10:
            violations.append("DIVERSIFICATION WARNING: International exposure is below the 10% safety threshold.")
            
        if not violations:
            return "✅ COMPLIANCE PASSED: The portfolio meets all safety and diversification guardrails."
        else:
            return "❌ COMPLIANCE FAILED:\n- " + "\n- ".join(violations)
    except Exception as e:
        return f"Guardrail check failed: {e}"

@mcp.tool()
def fetch_live_market_data(ticker: str) -> str:
    """
    Fetches real-time price and data for any public ticker (e.g., 'VTI', 'VXUS', 'AAPL') using yfinance.
    """
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        name = info.get("longName", ticker)
        er = info.get("fees_and_expenses") or info.get("expenseRatio", "N/A")
        
        return f"LIVE DATA for {ticker}:\n- Name: {name}\n- Current Price: ${price}\n- Expense Ratio: {er}"
    except Exception as e:
        return f"Failed to fetch data for {ticker}: {e}"

@mcp.tool()
def get_market_alternatives(category: str) -> str:
    """
    Returns the best-in-class public ETFs available via BrokerageLink for a given category.
    Categories: 'domestic_total', 'intl_total', 'bonds', 'emerging_markets'
    """
    market_data = {
        "domestic_total": [
            {"ticker": "VTI", "name": "Vanguard Total Stock Market", "er": 0.03},
            {"ticker": "FZROX", "name": "Fidelity Zero Total Market", "er": 0.00}
        ],
        "intl_total": [
            {"ticker": "VXUS", "name": "Vanguard Total Intl Stock", "er": 0.07},
            {"ticker": "FSPSX", "name": "Fidelity Intl Index", "er": 0.035}
        ],
        "bonds": [
            {"ticker": "BND", "name": "Vanguard Total Bond Market", "er": 0.03},
            {"ticker": "FXNAX", "name": "Fidelity U.S. Bond Index", "er": 0.025}
        ],
        "emerging_markets": [
            {"ticker": "VWO", "name": "Vanguard FTSE Emerging Markets", "er": 0.08},
            {"ticker": "SCHE", "name": "Schwab Emerging Markets Equity", "er": 0.11}
        ]
    }
    
    alts = market_data.get(category.lower())
    if not alts:
        return f"No specific recommendations for '{category}'. Generally, look for Vanguard (V) or Fidelity (F) index ETFs."
    
    res = [f"Top BrokerageLink Alternatives for {category.upper()}:"]
    for a in alts:
        res.append(f"- {a['ticker']}: {a['name']} (ER: {a['er']}%)")
    return "\n".join(res)

@mcp.tool()
def rebalance_portfolio(
    domestic_total: float,
    intl_total: float,
    gm_balance: float,
    truist_balance: float,
    lg_cap_share: float = 0.8,
    mid_cap_share: float = 0.1,
    sm_cap_share: float = 0.1,
    intl_total_share: float = 0.8,
    emerging_markets_share: float = 0.2
) -> str:
    """
    Calculates the mathematically optimal fund allocation across GM and Truist accounts.
    """
    targets = PortfolioTargets(
        domestic_total=domestic_total,
        intl_total=intl_total,
        lg_cap_share=lg_cap_share,
        mid_cap_share=mid_cap_share,
        sm_cap_share=sm_cap_share,
        intl_total_share=intl_total_share,
        emerging_markets_share=emerging_markets_share
    )
    
    balances = [
        {"account_name": "GM", "balance": gm_balance},
        {"account_name": "Truist", "balance": truist_balance}
    ]
    
    try:
        results = optimize_portfolio(targets, balances, persona_key="mcp_rebalance", persona_name="MCP Tool Execution")
        
        # Format a clean summary for the agent
        output = [f"Rebalance Complete for Total Portfolio: ${results['summary']['total_value']:,.2f}"]
        output.append(f"Aggregate Expense Ratio: {results['summary']['aggregate_expense_ratio']:.4f}%")
        output.append("\nALLOCATIONS:")
        
        for f in results['selected_funds']:
            output.append(f"- {f['name']} [{f['account_source'].upper()}]: ${f['allocated_dollars']:,.2f} ({f['shares']:,.2f} shares)")
            
        return "\n".join(output)
    except Exception as e:
        return f"Rebalancing failed: {e}"

# To host on a domain, we expose the MCP as a FastAPI app
app = mcp.http_app

if __name__ == "__main__":
    import uvicorn
    # Use the SSE transport for remote hosting
    uvicorn.run(app, host="0.0.0.0", port=8000)
