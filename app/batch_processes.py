import os
import time
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from app.schemas import Fund, AssetClassDetail
from app.vision import AzureDocumentClient

# Updated to match the new schema's allowed values
VALID_PARENTS = {"domestic", "international", "bonds", "cash"}
VALID_SUB_CLASSES = {"small_cap", "mid_cap", "large_cap", "emerging_markets", "total", None}

load_dotenv(override=True)

def is_valid_asset_class(ac: AssetClassDetail) -> bool:
    return ac.parent in VALID_PARENTS and ac.sub_class in VALID_SUB_CLASSES

def main():
    """
    Batch processes all documents.
    Usage: 
      python -m app.batch_processes --local [root_dir]
      python -m app.batch_processes [account_source]
    """
    args = sys.argv[1:]
    is_local = "--local" in args
    client = AzureDocumentClient()
    
    # Global consolidated file
    consolidated_file = Path("portfolio.json")
    all_funds = []
    if consolidated_file.exists():
        try:
            with open(consolidated_file, "r") as f:
                all_funds = json.load(f)
        except: pass

    if is_local:
        root_dir = Path(args[args.index("--local") + 1]) if len(args) > args.index("--local") + 1 else Path("input")
        print(f"Starting local batch extraction from: {root_dir}")
        
        files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.pdf"]:
            files.extend(list(root_dir.rglob(ext)))
        
        if not files:
            print(f"No image files found in '{root_dir}'.")
            return

        for index, file_path in enumerate(files):
            rel_path = file_path.relative_to(root_dir)
            # Use lowercase for folder names (e.g., "gm", "truist")
            account_source = rel_path.parts[0].lower() if len(rel_path.parts) > 1 else "auto"
            
            # Setup output directory: outputs/funds/[account_source]/
            output_dir = Path("outputs") / "funds" / account_source
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{file_path.stem}.json"

            # Skip if already processed
            if output_file.exists():
                print(f"[{index + 1}/{len(files)}] Skipping {file_path.name} (Already exists in outputs/)")
                continue

            print(f"[{index + 1}/{len(files)}] Processing {file_path.name}...")

            try:
                extracted_funds_data = client.extract_funds_from_file(file_path, account_source=account_source)
                processed_batch = process_fund_data(extracted_funds_data)
                
                # Save individual file
                with open(output_file, "w") as f:
                    json.dump([f.model_dump() for f in processed_batch], f, indent=2)
                
                # Update consolidated
                all_funds.extend([f.model_dump() for f in processed_batch])
                with open(consolidated_file, "w") as f:
                    json.dump(all_funds, f, indent=2)

                if index < len(files) - 1: time.sleep(1)
            except Exception as e:
                print(f"FAILED on {file_path}: {e}")

    else:
        # Azure Mode
        account_source = args[0] if args else "Auto"
        container_name = os.getenv("AZURE_STORAGE_CONTAINER", "prospectuses")
        print(f"Starting Azure batch extraction for '{account_source}'")

        try:
            container_client = client.blob_service_client.get_container_client(container_name)
            blobs = list(container_client.list_blobs())
        except Exception as e:
            print(f"Failed to list blobs: {e}")
            return

        for index, blob in enumerate(blobs):
            print(f"[{index + 1}/{len(blobs)}] Processing {blob.name}...")
            try:
                extracted_funds_data = client.extract_funds_from_blob(container_name, blob.name, account_source=account_source)
                processed_batch = process_fund_data(extracted_funds_data)
                
                all_funds.extend([f.model_dump() for f in processed_batch])
                with open(consolidated_file, "w") as f:
                    json.dump(all_funds, f, indent=2)
                
                if index < len(blobs) - 1: time.sleep(1)
            except Exception as e:
                print(f"FAILED on {blob.name}: {e}")

    print(f"\nExtraction complete. Consolidated data in {consolidated_file}")

def process_fund_data(extracted_funds_data):
    processed_funds = []
    for fund_data in extracted_funds_data:
        fund = None
        while fund is None:
            try:
                fund = Fund(**fund_data)
            except Exception as e:
                print(f"\n[!] SCHEMA VALIDATION ERROR for {fund_data.get('name', 'Unknown')}")
                if "name" not in fund_data or not fund_data["name"]:
                    fund_data["name"] = input("    Enter Fund Name: ").strip()
                
                parent = fund_data.get("asset_class", {}).get("parent")
                if parent not in VALID_PARENTS:
                    fund_data.setdefault("asset_class", {})["parent"] = input(f"    Enter parent {list(VALID_PARENTS)}: ").strip()
                
                sub = fund_data.get("asset_class", {}).get("sub_class")
                if sub not in VALID_SUB_CLASSES:
                    new_sub = input(f"    Enter sub-class {list(VALID_SUB_CLASSES)} (empty for None): ").strip()
                    fund_data.setdefault("asset_class", {})["sub_class"] = new_sub if new_sub else None

        while not is_valid_asset_class(fund.asset_class):
            new_parent = input(f"    Re-enter parent {list(VALID_PARENTS)}: ").strip()
            new_sub = input(f"    Re-enter sub-class {list(VALID_SUB_CLASSES)}: ").strip()
            fund.asset_class = AssetClassDetail(parent=new_parent, sub_class=new_sub if new_sub else None)

        while fund.expense_ratio is None:
            print(f"\n[!] WARNING: Missing expense ratio for {fund.name} ({fund.symbol}).")
            try:
                fund.expense_ratio = float(input("    Enter expense ratio: ").strip())
            except ValueError: pass
        processed_funds.append(fund)
    return processed_funds

if __name__ == "__main__":
    main()
