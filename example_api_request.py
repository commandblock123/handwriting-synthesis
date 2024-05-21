import requests
import os

url = "http://127.0.0.1:8080/synthesize_handwriting"

text = input("Text? ")

body = {
    "text": text,
    "style": 5,
    "bias": 0.75,
    "stroke_color": "black",
    "stroke_width": 1
}

response = requests.post(url=url, json=body)

print(response.status_code)
if response.status_code == 200:
    with open('generated_handwriting.svg', 'w', encoding='utf-8') as f:
        f.write(response.content.decode('utf-8'))
        print("SVG file saved.")
else:
    print(response.content)