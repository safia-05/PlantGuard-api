import requests

url = "http://127.0.0.1:8000/identify"
filename = "non.jpg"

with open(filename, "rb") as f:
    files = {"file": (filename, f, "image/jpeg")}   # force MIME type
    response = requests.post(url, files=files)
    print("Status:", response.status_code)
    print("Response:", response.json())