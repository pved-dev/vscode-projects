import os
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


ERPNEXT_URL = os.getenv("ERPNEXT_URL")
API_KEY = os.getenv("ERPNEXT_API_KEY")
API_SECRET = os.getenv("ERPNEXT_API_SECRET")

headers = {
    "Authorization": f"token {API_KEY}:{API_SECRET}"
}

def test_connection():
    response = requests.get(
        f"{ERPNEXT_URL}/api/method/frappe.auth.get_logged_user",
        headers=headers
    )
    
    if response.status_code == 200:
        print("✅ Connection successful!")
        print(f"Logged in as: {response.json()['message']}")
    else:
        print(f"❌ Connection failed: {response.status_code}")
        print(response.text)

def test_inventory():
    response = requests.get(
        f"{ERPNEXT_URL}/api/resource/Item?limit=5",
        headers=headers
    )
    
    if response.status_code == 200:
        items = response.json()["data"]
        print(f"\n✅ Found {len(items)} items in inventory:")
        for item in items:
            print(f"  - {item['name']}")
    else:
        print(f"❌ Inventory query failed: {response.status_code}")

if __name__ == "__main__":
    test_connection()
    test_inventory()