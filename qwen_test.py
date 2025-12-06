import os
import requests

api_key = os.environ.get("sk-b41ba2dcc59f40f797ea687366da74b5")
url = "https://api.qwen.cloud/v1/chat/completions"
headers = {"Authorization": f"Bearer {api_key}"}

data = {
    "model": "qwen-7b-chat",
    "messages": [{"role": "user", "content": "Refine this query: man riding bicycle"}]
}

response = requests.post(url, json=data, headers=headers)
print(response.json()["choices"][0]["message"]["content"])
