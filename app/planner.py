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

def find_cheapest_fund(funds, parent, sub_class):
    relevant = [
        f for f in funds 
        if f.asset_class.parent == parent and f.asset_class.sub_class == sub_class
    ]
    if not relevant:
        return None
    return min(relevant, key=lambda x: (x.expense_ratio or 1.0, x.name))

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
        f"- {f.name} ({f.asset_class.parent} {f.asset_class.sub_class}): ER {f.expense_ratio}%"
        for f in selected_funds
    ])
    
    sector_desc = "\n".join([f"- {s}: {w*100:.2f}%" for s, w in sorted(aggregate_sectors.items(), key=lambda x: x[1], reverse=True)])

    prompt = f"""
You are a Senior Portfolio Strategy Agent. I have mathematically optimized a portfolio using ONLY the funds available in my GM and Truist 401k plans.

MATH-OPTIMIZED PORTFOLIO (Best of Plan):
{portfolio_desc}

AGGREGATE EXPENSE RATIO: {total_er:.4f}%

SECTOR EXPOSURE:
{sector_desc}

STRATEGIC TASK:
1. Evaluate if any of these "Best of Plan" funds are actually "bad" (e.g. Expense Ratio > 0.05% or poor diversification).
2. BrokerageLink exists: I can buy ANY public ETF or Mutual Fund. 
3. Recommend specific BrokerageLink alternatives (e.g. VTI, VXUS, FZROX) if they significantly beat the plan options in cost or exposure.
4. Highlight any "Sector Craters" (industries I am missing) or "Sector Peaks" (over-concentration).

Provide a concise, professional "Executive Advisory" summary.
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

def optimize_portfolio(targets: PortfolioTargets, persona_key: str = "manual", persona_name: str = "Manual"):
    all_funds = load_all_funds()
    if not all_funds:
        print("No funds found in outputs/funds/. Run batch_processes first.")
        return

    summary_io = io.StringIO()
    selection_plan = [
        ("domestic", "large_cap", targets.domestic_total * targets.lg_cap_share),
        ("domestic", "mid_cap", targets.domestic_total * targets.mid_cap_share),
        ("domestic", "small_cap", targets.domestic_total * targets.sm_cap_share),
        ("international", "total", targets.intl_total * targets.intl_total_share),
        ("international", "emerging_markets", targets.intl_total * targets.emerging_markets_share),
    ]

    selected_portfolio = []
    total_er = 0.0
    aggregate_sectors = defaultdict(float)

    header = f"\nPLANNER STRATEGY: {persona_name.upper()}\n" + "="*50
    print(header)
    summary_io.write(header + "\n")

    for parent, sub_class, weight in selection_plan:
        if weight <= 0: continue
        best_fund = find_cheapest_fund(all_funds, parent, sub_class)
        if not best_fund:
            msg = f"[!] WARNING: No fund found for {parent} {sub_class}"
            print(msg)
            summary_io.write(msg + "\n")
            continue

        fund_copy = best_fund.model_copy()
        fund_copy.allocation_percent = weight * 100
        selected_portfolio.append(fund_copy)
        er = fund_copy.expense_ratio or 0.0
        total_er += (er * weight)

        if fund_copy.sectors:
            for sector, val in fund_copy.sectors.items():
                aggregate_sectors[sector] += (val * weight)

        line1 = f"{parent.upper()} {sub_class.upper()} ({weight*100:.1f}%)"
        line2 = f"  > {fund_copy.name}"
        line3 = f"  > Expense Ratio: {er:.4f}% | Source: {fund_copy.account_source}"
        for l in [line1, line2, line3]:
            print(l)
            summary_io.write(l + "\n")

    output_dir = Path("outputs/planner") / persona_key
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_file = output_dir / "plan.json"
    summary_file = output_dir / "summary.txt"

    result_data = {
        "summary": {
            "persona": persona_name,
            "aggregate_expense_ratio": total_er,
            "targets": targets.model_dump(),
            "sector_makeup": aggregate_sectors
        },
        "selected_funds": [f.model_dump() for f in selected_portfolio]
    }

    with open(plan_file, "w") as f:
        json.dump(result_data, f, indent=2)

    footer = f"\nPORTFOLIO AGGREGATE EXPENSE RATIO: {total_er:.4f}%\n" + "-"*50
    print(footer)
    summary_io.write(footer + "\n")

    makeup_header = "\nINDUSTRY MAKEUP (WEIGHTED):"
    print(makeup_header)
    summary_io.write(makeup_header + "\n")
    
    sorted_sectors = sorted(aggregate_sectors.items(), key=lambda x: x[1], reverse=True)
    for sector, weight in sorted_sectors:
        line = f"  {sector:25}: {weight*100:6.2f}%"
        print(line)
        summary_io.write(line + "\n")

    advisory_header = f"\nAI ADVISOR: {persona_name}\n" + "-"*50
    print(advisory_header)
    summary_io.write(advisory_header + "\n")
    advisory = get_agent_advisory(selected_portfolio, aggregate_sectors, total_er)
    print(advisory)
    summary_io.write(advisory + "\n")

    with open(summary_file, "w") as f:
        f.write(summary_io.getvalue())
    print(f"SAVED TO: {output_dir}/")

from app.personas import PERSONAS, get_targets_for_persona

def main():
    parser = argparse.ArgumentParser(description="AI Portfolio Planner.")
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
        print("RUNNING MULTI-AGENT PLANNER COMPARISON")
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
