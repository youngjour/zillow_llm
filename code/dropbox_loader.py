import os
from dotenv import load_dotenv
import requests

# Load .env file
load_dotenv()

DROPBOX_BASE = os.getenv("DROPBOX_BASE")
DROPBOX_PREFIX = os.getenv("DROPBOX_PREFIX")
DROPBOX_RLKEY = os.getenv("DROPBOX_RLKEY")

def get_dropbox_download_url(filename):
    """Construct Dropbox download URL for a file inside a folder."""
    return f"{DROPBOX_BASE}/{filename}?rlkey={DROPBOX_RLKEY}&dl=1"

def download_file(filename, save_dir="dataset"):
    """Download file from Dropbox if not already downloaded."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        url = get_dropbox_download_url(filename)
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Saved to {filepath}")
        else:
            raise Exception(f"Failed to download {filename}. HTTP {response.status_code}")
    else:
        print(f"File already exists: {filepath}")
    
    return filepath