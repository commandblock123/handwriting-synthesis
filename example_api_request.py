import requests
import os

url = ""

text = "this is an example"

body = {
    "text": text,
    "style": 5,
    "bias": 0.75,
    "stoke_color": "black",
    "stroke_width": 1
}

response = requests.post(url=url, data=body)

print(response.status_code)
print(response.content)

# os.startfile("img\\give_up.svg")