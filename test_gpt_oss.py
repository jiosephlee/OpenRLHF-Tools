import urllib.request
import json
url = "https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json"
req = urllib.request.Request(url)
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read())
print(json.dumps(data, indent=2))
