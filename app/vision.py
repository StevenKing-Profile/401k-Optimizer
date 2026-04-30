import os
import json
import base64
import io
from pathlib import Path
from PIL import Image
from app.schemas import Fund
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI


class AzureDocumentClient:
    def __init__(self):
        # Azure Storage Setup (Entra ID Authorization)
        self.credential = DefaultAzureCredential()
        account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
        
        if not account_url:
            raise ValueError("AZURE_STORAGE_ACCOUNT_URL environment variable is missing or empty.")
            
        self.blob_service_client = BlobServiceClient(account_url, credential=self.credential)

        # Azure OpenAI Setup (Static Key Authorization)
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is missing.")

        self.openai_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-08-01-preview",
            api_key=api_key
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

    def extract_funds_from_blob(self, container_name: str, blob_name: str, account_source: str) -> list[dict]:
        """Downloads from Azure and processes."""
        blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        document_bytes = blob_client.download_blob().readall()
        return self.extract_funds(document_bytes, blob_name, account_source)

    def extract_funds_from_file(self, file_path: Path, account_source: str) -> list[dict]:
        """Reads local file and processes."""
        with open(file_path, "rb") as f:
            document_bytes = f.read()
        return self.extract_funds(document_bytes, file_path.name, account_source)

    def extract_funds(self, document_bytes: bytes, display_name: str, account_source: str) -> list[dict]:
        """
        The core logic: Slices bytes, sends to GPT-4o, and parses JSON.
        """
        # Slice the image into overlapping vertical segments
        img = Image.open(io.BytesIO(document_bytes))
        width, height = img.size
        
        segment_height = (height // 2) + int(height * 0.1)
        coords = [
            (0, 0, width, segment_height),
            (0, height - segment_height, width, height)
        ]
        
        image_contents = []
        for i, box in enumerate(coords):
            crop = img.crop(box)
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            base64_crop = base64.b64encode(buf.getvalue()).decode('utf-8')
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_crop}", "detail": "high"}
            })

        extract_prompt = f"""
### SYSTEM ROLE
You are a High-Fidelity Financial Data Extraction Engine. Your objective is 100% numeric accuracy.

### PHASE 1: SPATIAL TRANSCRIPTION (Markdown)
1. DETAILS TABLE: Transcribe 'Exp Ratio (Gross)', 'Exp Ratio (Net)', and 'NAV'.
2. SECTORS: 3-column grid [ICON | PORTFOLIO % | INDEX %]. Portfolio Weight is the FIRST numeric value.
3. REGIONAL DIVERSIFICATION: 
   - Only transcribe this table if the document indicates it is an 'International', 'Global', or 'ex-US' fund. 
   - If it is a Domestic/US-only fund, skip this table and return null for regions in JSON.
   - If transcribing, ensure 'Portfolio Weight' values sum to ~100%.

### PHASE 2: LOGIC & VERIFICATION
1. Expense Ratio: Transcribe both 'Exp Ratio (Net)' and 'Exp Ratio (Gross)'.
2. Numeric Conversion: 
   - SECTORS/REGIONS: Convert percentages (e.g., 15.73%) to 4-place decimals (0.1573).
   - EXPENSE RATIO: Use the 'Exp Ratio (Net)' value as a float (e.g., 0.75% becomes 0.75). Do NOT return a dictionary.
3. Sector Mapping: [Financials, Information Technology, Industrials, Consumer Discretionary, Health Care, Materials, Consumer Staples, Energy, Communication Services].

### PHASE 3: JSON
Return ONLY the raw JSON object.
{{
  "name": "{display_name.split('.')[0]}",
  "symbol": null,
  "expense_ratio": float,
  "nav": decimal,
  "asset_class": {{ 
    "parent": "domestic"|"international", 
    "sub_class": "small_cap"|"mid_cap"|"large_cap"|"emerging_markets"|"total" 
  }},
  "NOTE": "For International funds, 'Global ex-US', 'International Index', or 'Total International' MUST be mapped to 'total' sub_class, NOT 'large_cap'."
  "sectors": {{ "Label": decimal }},
  "regions": {{ "Label": decimal }}
}}
"""

        response = self.openai_client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a technical OCR expert using high-resolution segments."},
                {"role": "user", "content": [{"type": "text", "text": extract_prompt}, *image_contents]}
            ],
            temperature=0.0
        )

        choice = response.choices[0]
        full_text = choice.message.content or ""

        # Extract JSON from the multi-segment response
        raw_json = ""
        if "```json" in full_text:
            raw_json = full_text.split("```json")[1].split("```")[0].strip()
        else:
            try:
                start = full_text.find("{")
                end = full_text.rfind("}") + 1
                if start != -1 and end != -1:
                    raw_json = full_text[start:end]
            except Exception:
                pass

        if not raw_json:
            print(f"[!] ERROR: Could not find JSON in response. Full Output:\n{full_text[:500]}...")
            return []

        try:
            extracted_data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            print(f"[!] ERROR: Failed to parse JSON: {e}")
            return []

        if isinstance(extracted_data, dict) and "funds" in extracted_data:
            extracted_data = extracted_data["funds"]

        if isinstance(extracted_data, dict):
            extracted_data = [extracted_data]

        if not isinstance(extracted_data, list):
            return []

        # Normalization
        for item in extracted_data:
            item["account_source"] = account_source

            # Rescue identity if nested
            if "fund_identity" in item:
                identity = item.pop("fund_identity")
                if isinstance(identity, dict):
                    for k in ["name", "symbol"]:
                        if k in identity and k not in item:
                            item[k] = identity[k]

            # Rescue expense_ratio if nested in a dict
            if isinstance(item.get("expense_ratio"), dict):
                er_dict = item["expense_ratio"]
                item["expense_ratio"] = er_dict.get("net") or er_dict.get("gross") or next((v for v in er_dict.values() if isinstance(v, (int, float))), None)

            # Ensure numeric types for sectors and regions, removing Nulls
            for field in ["sectors", "regions"]:
                if field in item:
                    if isinstance(item[field], dict):
                        # Filter out None values and ensure floats
                        cleaned = {}
                        for k, v in item[field].items():
                            try:
                                if v is not None:
                                    cleaned[k] = float(v)
                            except (ValueError, TypeError):
                                pass
                        item[field] = cleaned if cleaned else None
                    elif isinstance(item[field], list):
                        flattened = {}
                        for entry in item[field]:
                            if isinstance(entry, dict):
                                keys = list(entry.keys())
                                if len(keys) >= 2:
                                    try:
                                        label = str(entry[keys[0]])
                                        val = entry[keys[1]]
                                        if val is not None:
                                            flattened[label] = float(val)
                                    except (ValueError, TypeError):
                                        pass
                        item[field] = flattened if flattened else None
                    else:
                        item[field] = None

        return extracted_data