import requests

key = ""  # your real key here

r = requests.get(
    f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
)
data = r.json()
for m in data.get("models", []):
    if "generateContent" in m.get("supportedGenerationMethods", []):
        print(m["name"])