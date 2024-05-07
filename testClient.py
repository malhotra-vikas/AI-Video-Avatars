import requests
import json

# Define the URL
url = "https://90kc8.apps.beam.cloud"
#url = 'https://90kc8-663979929575710009c632d5.apps.beam.cloud'
audio = "https://1010public.s3.us-east-2.amazonaws.com/Male.wav"

# Define the headers
headers = {
  "Accept": "*/*",
  "Accept-Encoding": "gzip, deflate",
  "Authorization": "Basic ZDRlNGNiNzhiYzMxYmU0YmViYWFiZDIyNWY4MDhkODY6NWVmNjljZmRhYWM4YTZjYWMxODFlMTcyM2VjMWUwYjM=",
  "Connection": "keep-alive",
  "Content-Type": "application/json"
}

# Define the payload
payload = {
    #"audio": "./data/audio.wav",
    "audio": audio,
    "image": "./data/man-1.png",
    "character": "Vikas-1",
    "test": True, 
#    "enhancer": "gfpgan",
#    "background-enhancer": True,
    "preprocess": "crop",
    "still": False
}

# Make the POST request
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Check the response
print(response.status_code)
print(response.text)
