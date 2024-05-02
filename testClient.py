import requests
import json

# Define the URL
url = 'https://couzt.apps.beam.cloud' 
#url = 'https://couzt-661545fab84d890007ce9428.apps.beam.cloud'
audio = "https://1010public.s3.us-east-2.amazonaws.com/bible_quotes/fd0162c0-026f-4428-b113-78398e6f426f.mp3"

# Define the headers
headers = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate',
    'Authorization': 'Basic OGUyOTQyNGQ2YWE4MWNiZDZlNmZhYmQzNjdkNTNkYzA6YjNhMTEzNDhmMTQ0M2FmZGJhZjM4ODlhOGExYTdhZDQ=',
    'Connection': 'keep-alive',
    'Content-Type': 'application/json'
}

# Define the payload
payload = {
    #"audio": "./data/audio.wav",
    "audio": audio,
    "image": "./data/jesus.png",
    "character": "Jesus",
    "test": True # Set to True when Testing and False for Prod Runs
}

# Make the POST request
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Check the response
print(response.status_code)
print(response.text)
