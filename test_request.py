import urllib.request
import json

"""
Send a test completion request to the local VDLM server
"""

def send_request():
    url = "http://localhost:8000/completions"
    payload = {
        "model": "vdlm-v1",
        "prompt": "How can I solve 2x=5?",
        "max_tokens": 16,
        "block_length": 4,
    }

    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req) as response:
            res_data = response.read().decode("utf-8")
            print(f"Status Code: {response.getcode()}")
            print(json.dumps(json.loads(res_data), indent=4))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        error_body = e.read().decode("utf-8")
        try:
            print(json.dumps(json.loads(error_body), indent=4))
        except:
            print(error_body)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    send_request()
