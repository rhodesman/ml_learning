import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

API_KEY = os.getenv("COINBASE_API_KEY")
API_SECRET = os.getenv("COINBASE_API_SECRET")

# Test API Key with an account endpoint
url = "https://api.coinbase.com/api/v3/brokerage/accounts"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)
print("Status Code:", response.status_code)
print("Response Text:", response.text)