import json
import io
from pathlib import Path
from collections import defaultdict
import argparse
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from app.schemas import Fund, PortfolioTargets

load_dotenv()

def load_all_funds():
    funds = []
    output_path = Path("outputs/funds")
    if not output_path.exists():
        return []
        
    for json_file in output_path.rglob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                for item in data:
                    funds.append(Fund(**item))
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return funds

def get_all_account_options(funds, parent, sub_class):
    """Returns all funds for a category, sorted by cost."""
    relevant = [
        f for f in funds 
        if f.asset_class.parent == parent and f.asset_class.sub_class == sub_class
    ]
    return sorted(relevant, key=lambda x: (x.expense_ratio or 1.0, x.name))

def get_agent_advisory(selected_funds, aggregate_sectors, total_er):
    """
    Uses GPT-4o to provide a qualitative analysis of the math-optimized portfolio.
    """
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-08-01-preview",
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

    portfolio_desc = "\n".join([
        f"- {f.name} ({f.asset_class.parent} {f.asset_class.sub_class}) in {f.account_source.upper()}: ER {f.expense_ratio}%"
        for f in selected_funds
    ])
    
    sector_desc = "\n".join([f"- {s}: {w*100:.2f}%" for s, w in sorted(aggregate_sectors.items(), key=lambda x: x[1], reverse=True)])

    prompt = f"""
You are a Senior Portfolio Strategy Agent. I have mathematically optimized a portfolio using funds from two different 401k plans: GM and TRUIST.

MATH-OPTIMIZED PORTFOLIO:
{portfolio_desc}

AGGREGATE EXPENSE RATIO: {total_er:.4f}%

SECTOR EXPOSURE:
{sector_desc}

STRATEGIC TASK:
1. **Account Rationale:** Explain WHY specific funds were chosen for GM vs. TRUIST (e.g., "The S&P 500 index in GM was selected because it's significantly cheaper than the Truist option, while the Mid-Cap exposure was placed in Truist to balance the account limits").
2. **Quality Check:** Evaluate if any of these "Best of Plan" funds are actually "bad" (ER > 0.05% or poor diversification).
3. **BrokerageLink:** Since I can buy ANY ETF (VTI, VXUS, etc.) via BrokerageLink, identify which plan funds should be replaced first.
4. **Sector Analysis:** Highlight any missing sectors or over-concentrations.

Provide a concise, professional "Executive Advisory" summary. Use clear headings.
"""

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Agent Advisory Unavailable: {e}"

def optimize_portfolio(targets: PortfolioTargets, account_balances: list[dict], persona_key: str = "manual", persona_name: str = "Manual"):
    all_funds = load_all_funds()
    if not all_funds:
        print("No funds found in outputs/funds/. Run batch_processes first.")
        return

    # Work with absolute Dollars
    total_portfolio_value = sum(a["balance"] for a in account_balances)
    if total_portfolio_value == 0:
        print("Total balance is zero.")
        return
    
    account_dollars = {a["account_name"].lower(): a["balance"] for a in account_balances}
    remaining_dollars = account_dollars.copy()

    # Define target dollars for each asset class
    selection_plan = [
        ("domestic", "large_cap", targets.domestic_total * targets.lg_cap_share),
        ("domestic", "mid_cap", targets.domestic_total * targets.mid_cap_share),
        ("domestic", "small_cap", targets.domestic_total * targets.sm_cap_share),
        ("international", "total", targets.intl_total * targets.intl_total_share),
        ("international", "emerging_markets", targets.intl_total * targets.emerging_markets_share),
    ]

    selected_portfolio = []
    total_er_dollars = 0.0
    aggregate_sectors = defaultdict(float)

    summary_io = io.StringIO()
    header = f"\nREBALANCER STRATEGY: {persona_name.upper()} (MULTI-ACCOUNT)\n" + "="*60
    print(header)
    summary_io.write(header + "\n")

    # Greedy Allocation by Asset Class
    for parent, sub_class, target_weight in selection_plan:
        if target_weight <= 0: continue
        
        target_dollars = target_weight * total_portfolio_value
        remaining_to_fill = target_dollars
        
        # Get options across all accounts, sorted by cost
        options = get_all_account_options(all_funds, parent, sub_class)
        
        for fund in options:
            source = fund.account_source.lower()
            if source not in remaining_dollars or remaining_dollars[source] <= 0:
                continue
            
            # Allocate from this account
            amount_to_take = min(remaining_to_fill, remaining_dollars[source])
            
            # Calculate shares
            nav = fund.nav if (fund.nav and fund.nav > 0) else 1.0 # Fallback to 1.0 if NAV missing
            shares = amount_to_take / nav
            
            fund_copy = fund.model_copy()
            fund_copy.allocation_percent = (amount_to_take / total_portfolio_value) * 100
            
            # Add custom attributes for the result (shares and dollars)
            fund_dict = fund_copy.model_dump()
            fund_dict["shares"] = round(shares, 4)
            fund_dict["allocated_dollars"] = round(amount_to_take, 2)
            selected_portfolio.append(fund_dict)
            
            er = fund_copy.expense_ratio or 0.0
            total_er_dollars += (er * (amount_to_take / 100.0)) # ER is usually a % (e.g. 0.05 means 0.05%)
            
            if fund_copy.sectors:
                for sector, val in fund_copy.sectors.items():
                    aggregate_sectors[sector] += (val * (amount_to_take / total_portfolio_value))
            
            remaining_to_fill -= amount_to_take
            remaining_dollars[source] -= amount_to_take
            
            line = f"{parent.upper()} {sub_class.upper()} -> ${amount_to_take:,.2f} ({shares:,.2f} shares) in {fund_copy.name} [{source.upper()}]"
            print(line)
            summary_io.write(line + "\n")
            
            if remaining_to_fill <= 0.01:
                break
        
        if remaining_to_fill > 1.0: # Allow for small floating point noise
            msg = f"[!] ALERT: Could not fully allocate {parent} {sub_class}. Missing ${remaining_to_fill:,.2f}"
            print(msg)
            summary_io.write(msg + "\n")

    # Handle Free Cash
    for source, remaining in remaining_dollars.items():
        if remaining > 0.1:
            line = f"CASH REMAINING -> ${remaining:,.2f} in {source.upper()}"
            print(line)
            summary_io.write(line + "\n")
            selected_portfolio.append({
                "name": "Cash / Sweep",
                "symbol": "CASH",
                "allocation_percent": (remaining / total_portfolio_value) * 100,
                "allocated_dollars": round(remaining, 2),
                "shares": round(remaining, 2), # 1:1 for cash
                "expense_ratio": 0.0,
                "account_source": source,
                "asset_class": {"parent": "cash", "sub_class": None}
            })

    total_er_pct = (total_er_dollars / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0.0

    # Save results
    output_dir = Path("outputs/rebalancer") / persona_key
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_file = output_dir / "plan.json"
    summary_file = output_dir / "summary.txt"

    result_data = {
        "summary": {
            "persona": persona_name,
            "total_value": total_portfolio_value,
            "aggregate_expense_ratio": total_er_pct,
            "targets": targets.model_dump(),
            "account_dollars": account_dollars,
            "sector_makeup": aggregate_sectors
        },
        "selected_funds": selected_portfolio
    }
    with open(plan_file, "w") as f:
        json.dump(result_data, f, indent=2)

    footer = f"\nPORTFOLIO AGGREGATE EXPENSE RATIO: {total_er_pct:.4f}%\n" + "-"*60
    print(footer)
    summary_io.write(footer + "\n")
    
    # Pass simplified Fund objects back to advisory for readability
    advisory_funds = [Fund(**f) for f in selected_portfolio if f["symbol"] != "CASH"]
    advisory = get_agent_advisory(advisory_funds, aggregate_sectors, total_er_pct)
    print("\nAI ADVISORY:\n" + advisory)
    summary_io.write("\nAI ADVISORY:\n" + advisory + "\n")

    with open(summary_file, "w") as f:
        f.write(summary_io.getvalue())
    
    print(f"SAVED TO: {output_dir}/")
    return result_data



from app.personas import PERSONAS, get_targets_for_persona

def main():
    parser = argparse.ArgumentParser(description="AI Portfolio Rebalancer.")
    parser.add_argument("--persona", type=str, choices=list(PERSONAS.keys()))
    parser.add_argument("--domestic", type=float, default=0.60)
    parser.add_argument("--intl", type=float, default=0.40)
    parser.add_argument("--large", type=float, default=0.80)
    parser.add_argument("--mid", type=float, default=0.10)
    parser.add_argument("--small", type=float, default=0.10)
    parser.add_argument("--intl_total", type=float, default=0.80)
    parser.add_argument("--emerging", type=float, default=0.20)
    args = parser.parse_args()

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-08-01-preview",
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    if args.persona:
        run_optimization_for_persona(args.persona, client)
    else:
        print("\n" + "#"*60)
        print("RUNNING MULTI-AGENT REBALANCER COMPARISON")
        print("#"*60)
        for persona_key in PERSONAS.keys():
            run_optimization_for_persona(persona_key, client)

def run_optimization_for_persona(persona_key, client):
    print(f"\n[AGENT] Invoking Expert Persona: {PERSONAS[persona_key].name}...")
    try:
        targets = get_targets_for_persona(persona_key, client)
        print(f"[AGENT] Philosophy: {PERSONAS[persona_key].philosophy}")
        optimize_portfolio(targets, persona_key=persona_key, persona_name=PERSONAS[persona_key].name)
    except Exception as e:
        print(f"[!] Error running persona {persona_key}: {e}")

if __name__ == "__main__":
    main()
