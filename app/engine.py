import numpy as np
from scipy.optimize import linprog
from app.schemas import Fund, PortfolioTargets


def solve_portfolio_optimization(funds: list[Fund], balances: dict[str, float], targets: PortfolioTargets):
    """
    Uses Linear Programming to find the absolute cheapest way to hit targets.
    """
    total_value = sum(balances.values())

    # 1. Coefficients for Objective Function (minimize the sum of (Allocation * Expense Ratio))
    costs = [f.expense_ratio if f.expense_ratio is not None else 0.50 for f in funds]

    # 2. Equality Constraints (The Targets in dollars)
    target_map = {
        "lg_cap": total_value * targets.lg_cap_weight,
        "mid_cap": total_value * targets.mid_cap_weight,
        "sm_cap": total_value * targets.sm_cap_weight,
        "intl": total_value * targets.intl_total,
    }

    A_eq = []
    b_eq = []

    for asset_class, target_dollars in target_map.items():
        # Create a row of 1s and 0s. 1 if the fund is in this asset class.
        row = [1 if f.asset_class == asset_class else 0 for f in funds]
        A_eq.append(row)
        b_eq.append(target_dollars)

    # 3. Inequality Constraints (Account Limits) - can't spend more than I have
    A_ub = []
    b_ub = []

    for account_name, account_balance in balances.items():
        # Create a row of 1s if the fund belongs to this specific account
        # Note: We'll need to add 'account_source' to our Fund schema
        row = [1 if getattr(f, 'account_source', '') == account_name else 0 for f in funds]
        A_ub.append(row)
        b_ub.append(account_balance)

    # 4. Run the Solver
    res = linprog(costs, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, method='highs')

    if res.success:
        return {funds[i].name: round(res.x[i], 2) for i in range(len(funds))}
    else:
        return {"error": "No valid allocation found. Check if your targets exceed account balances."}