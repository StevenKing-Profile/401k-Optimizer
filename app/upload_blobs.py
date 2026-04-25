import os
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

def upload_images(directory="input", container_name="prospectuses"):
    account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
    if not account_url:
        print("Error: AZURE_STORAGE_ACCOUNT_URL not set in .env")
        return

    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url, credential=credential)
    container_client = blob_service_client.get_container_client(container_name)

    # Ensure container exists
    try:
        container_client.create_container()
        print(f"Created container: {container_name}")
    except Exception:
        pass

    base_path = Path(directory)
    for file_path in base_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".pdf"]:
            # Use relative path as blob name to preserve folder structure
            blob_name = str(file_path.relative_to(base_path))
            blob_client = container_client.get_blob_client(blob_name)
            
            print(f"Uploading {file_path} as {blob_name}...")
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

    print("Upload complete.")

if __name__ == "__main__":
    upload_images()
