import requests

url = "https://api.synclabs.so/lipsync"

# Initiates the animation
# Choice of three models 
    # wav2lip++ (Free to use. Lower quality)
    # sync-1.5.0
    # sync-1.6.0 (Best model in terms of human like output on the video)

def initiateVideoanimation(audio, video, model, callback, synergize=True):
    payload = {
        "audioUrl": audio,
        "videoUrl": video,
        "synergize": synergize,
        "model": model,
        "webhookUrl": callback,

    }
    headers = {
        "x-api-key": "a7d0f170-9e49-438e-838f-8990879cfa54",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    print(response.text)

# Given a task Id, fetches the output
def fetchAnimatedVideo(taskid):

    url = url+taskid

    headers = {"x-api-key": "a7d0f170-9e49-438e-838f-8990879cfa54"}

    response = requests.request("GET", url, headers=headers)

    if (response):
        print(response.text)
        animatedVideo = response.url
        jobStatus = response.status
    
