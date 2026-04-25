from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Literal

class AssetClassDetail(BaseModel):
    model_config = ConfigDict(frozen=True)
    parent: Literal["domestic", "international", "bonds", "cash"]
    sub_class: Optional[Literal["small_cap", "mid_cap", "large_cap", "emerging_markets", "total"]] = None

class Fund(BaseModel):
    symbol: Optional[str] = None
    name: str
    allocation_percent: float = Field(default=0.0, ge=0, le=100)
    expense_ratio: Optional[float] = None
    asset_class: AssetClassDetail  # Nested hierarchy
    account_source: Optional[str] = None

    # Deep Diversification
    nav: Optional[float] = None
    sectors: Optional[Dict[str, float]] = None
    regions: Optional[Dict[str, float]] = None

class PortfolioTargets(BaseModel):
    domestic_total: float = 0.60
    intl_total: float = 0.40

    # Weights WITHIN the groups (should sum to 1.0 within their parent)
    # Domestic breakdown
    lg_cap_share: float = 0.80  # e.g. 80% of the domestic 60%
    mid_cap_share: float = 0.10
    sm_cap_share: float = 0.10

    # International breakdown
    intl_total_share: float = 0.80 # e.g. 80% of the intl 40%
    emerging_markets_share: float = 0.20