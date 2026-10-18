import json
import urllib.request

LLAMA_URL = "http://172.17.0.1:8081/v1/chat/completions"

def generate_maintenance_advice(predictions):
    prompt = f"""
    You are TrustGuard, an industrial maintenance assistant.

    Nitrogen generator predictions:
    {json.dumps(predictions)}

    Write exactly TWO short sentences:
    1. Summarize the failure risk using percentages.
    2. Recommend one practical monitoring action.

    Do not repeat all prediction values.
    Do not invent sensor data.
    Do not recommend shutting down equipment.
    """

    payload = {
        "model": "granite-small",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.2
    }

    request = urllib.request.Request(
        LLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)

    return result["choices"][0]["message"]["content"]