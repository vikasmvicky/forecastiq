import requests
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(".env"))
api_key = os.getenv("GROQ_API_KEY", "")

print(f"Key loaded: {api_key[:15]}...")
print(f"Key length: {len(api_key)}")

if not api_key:
    print("ERROR: No API key found in .env")
else:
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "temperature": 0.3, "max_tokens": 50, "messages": [{"role": "user", "content": "say ok"}]},
            timeout=10
        )
        print(f"Status: {r.status_code}")
        print(r.text[:300])
    except Exception as e:
        print(f"Error: {e}")