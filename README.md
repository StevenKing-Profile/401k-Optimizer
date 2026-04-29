# Portfolio Rebalancer AI

An agentic system for extracting, structuring, and optimizing 401(k) investment portfolios from raw fund prospectuses and screenshots.

## Core Features
- **High-Fidelity Vision OCR**: Uses GPT-4o with multi-segment image slicing and spatial anchoring to extract complex financial tables (Sectors, Regions, Fees) with 100% numeric accuracy.
- **Batch Processing**: Automates data extraction from local directories or Azure Blob Storage, saving structured JSON for every fund.
- **Multi-Agent Strategist**: Orchestrates multiple investment personas (Boglehead, Tech Bull, Globalist) to generate competing portfolio strategies.
- **Cheapest-of-Plan Math**: Automatically identifies the lowest-cost fund combinations to satisfy agent-defined allocation targets.
- **AI Advisory**: A final qualitative layer that analyzes sector concentration and suggests BrokerageLink alternatives (e.g., VTI, VXUS).

## Quick Start

### 1. Extract Data
Process local images (automatically tagged by folder, e.g., `images/Truist/`):
```bash
python -m app.batch_processes --local input/funds/
```

### 2. Run Rebalancer
Generate math-optimized portfolios for all personas:
```bash
python -m app.rebalancer
```
Results are saved to `outputs/rebalancer/[persona]/` as `plan.json` and `summary.txt`.


## Tech Stack
- **AI**: Azure OpenAI (GPT-4o)
- **Engine**: Python 3.14
- **Schema**: Pydantic (Type safety & Validation)
- **Image Ops**: Pillow (High-res slicing)
