from fastapi import FastAPI
from fastmcp import FastMCP
from app.schemas import Fund, PortfolioTargets
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Portfolio Optimizer")

@app.get("/ping")
def ping():
    return {"status": "alive", "version": "3.14"}

@app.post("/audit")
def audit_portfolio(current_holdings: List[Fund], targets: PortfolioTargets):
    """
    Analyzes current 401k holdings against targets and identifies
    'Poor Choices' that should be moved to BrokerageLink.
    """
    report = f"Analyzing {len(current_holdings)} funds against a {targets.domestic_total * 100}/{targets.intl_total * 100} split."

    for fund in current_holdings:
        if fund.expense_ratio and fund.expense_ratio > 0.10:
            report += f"\n- ALERT: {fund.name} ({fund.symbol}) has a high fee: {fund.expense_ratio}%"

    return report

mcp = FastMCP.from_fastapi(app)

if __name__ == "__main__":
    import uvicorn
    # Use the string-path to the app for proper module loading
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)