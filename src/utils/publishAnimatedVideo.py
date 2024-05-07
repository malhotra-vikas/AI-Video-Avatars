import requests
import json

prod_url = "http://64.23.252.7/add_new_video/2df3c7452e1c0eb9b6bf7e6965782323"
#test_url = "http://64.23.252.7/add_new_test_video/2df3c7452e1c0eb9b6bf7e6965782323"

def publishVideo(videoPath, videoName):
    print("Sending the video to the Remote Server", videoPath)    

    payload = {}

    files=[   
        (
        'file',(
            videoName,open(
                videoPath,'rb'
                ),'video/mp4'
                )
        )
    ]
    headers = {}
    url = ""

#    response = requests.request("POST", prod_url, headers=headers, data=payload, files=files)
    print("Skipping Remote Publish")

    print("Publish Video Respnse is ")
