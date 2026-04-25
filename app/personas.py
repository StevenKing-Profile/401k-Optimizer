import json
from typing import Dict
from app.schemas import PortfolioTargets

class InvestmentPersona:
    def __init__(self, name: str, philosophy: str, prompt_logic: str):
        self.name = name
        self.philosophy = philosophy
        self.prompt_logic = prompt_logic

PERSONAS = {
    "boglehead": InvestmentPersona(
        name="The Boglehead",
        philosophy="Passive indexing, lowest possible costs, and market-cap weighting.",
        prompt_logic="Aim for a classic 60/40 US/Intl split. Within US, use 80/10/10 for Lg/Mid/Small. Within Intl, use 80/20 for Total/Emerging."
    ),
    "techbull": InvestmentPersona(
        name="The Tech Bull",
        philosophy="Focus on where the innovation is: US Large Cap Growth.",
        prompt_logic="Aim for 90% Domestic, 10% Intl. Within US, put 95% into Large Cap. Ignore Small-cap and Emerging Markets."
    ),
    "globalist": InvestmentPersona(
        name="The Globalist",
        philosophy="The US is overvalued; growth will come from International and Emerging markets.",
        prompt_logic="Aim for 30% Domestic, 70% Intl. Within Intl, put 40% into Emerging Markets."
    )
}

def get_targets_for_persona(persona_key: str, agent_client) -> PortfolioTargets:
    """
    Uses Azure OpenAI to interpret the investment philosophy.
    """
    persona = PERSONAS.get(persona_key.lower())
    if not persona:
        raise ValueError(f"Unknown persona: {persona_key}")

    prompt = f"""
    You are an investment expert following this philosophy: {persona.philosophy}
    
    Task: Convert the following strategy into a JSON PortfolioTargets object:
    Strategy: {persona.prompt_logic}
    
    Required JSON Structure:
    {{
      "domestic_total": float, 
      "intl_total": float,
      "lg_cap_share": float, 
      "mid_cap_share": float,
      "sm_cap_share": float,
      "intl_total_share": float,
      "emerging_markets_share": float
    }}
    
    Ensure all domestic shares sum to 1.0 and all intl shares sum to 1.0.
    Return ONLY raw JSON.
    """
    
    response = agent_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    data = json.loads(response.choices[0].message.content)
    return PortfolioTargets(**data)
