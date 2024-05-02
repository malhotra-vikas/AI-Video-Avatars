from src.utils.publishAnimatedVideo import publishVideo
from animateVideoSyncLabs import initiateVideoanimation

file_path = "/Users/vikas/builderspace/quickstart/verses.csv"


# Replace these variables with your actual file URLs and desired model
audio_url = ""
video_url = "https://1010public.s3.us-east-2.amazonaws.com/asian.mp4"
model = "sync-1.6.2"  # Choosing the best model as per your description
#callback_url = "https://webhook.site/9e3f9c23-967e-4365-94d4-1cd42f1938f6"
callback_url = "https://webhook.site/ffd6be0c-9609-40c8-9710-d965fdab9b67"

"""

The WebHook Response will look like this. The Animated Video is "videoUrl"
{
  "result": {
    "id": "64ab525c-bfab-4701-b571-0d2efc20add3",
    "status": "COMPLETED",
    "videoUrl": "https://synchlabs-public.s3.amazonaws.com/lip-sync-jobs/2b1db764-0892-4cc0-97fd-276d7cd21ad0/64ab525c-bfab-4701-b571-0d2efc20add3/result.mp4",
    "originalVideoUrl": "https://1010public.s3.us-east-2.amazonaws.com/asian.mp4",
    "originalAudioUrl": "https://1010public.s3.us-east-2.amazonaws.com/bible_quotes/ceae8ce9-b546-4070-9f38-dab916235022.mp3",
    "synergize": true,
    "creditsDeducted": 261,
    "webhookUrl": "https://webhook.site/9e3f9c23-967e-4365-94d4-1cd42f1938f6",
    "message": "We have recently updated the response format for the /lipsync endpoint. This change is to ensure consistency across all endpoints. As a result of this change, the following properties have been deprecated and will be removed soon: (url, original_video_url, original_audio_url, credits_deducted). Please update your code to use the new properties: (videoUrl, originalVideoUrl, originalAudioUrl, creditsDeducted).",
    "url": "https://synchlabs-public.s3.amazonaws.com/lip-sync-jobs/2b1db764-0892-4cc0-97fd-276d7cd21ad0/64ab525c-bfab-4701-b571-0d2efc20add3/result.mp4",
    "original_video_url": "https://1010public.s3.us-east-2.amazonaws.com/asian.mp4",
    "original_audio_url": "https://1010public.s3.us-east-2.amazonaws.com/bible_quotes/ceae8ce9-b546-4070-9f38-dab916235022.mp3",
    "credits_deducted": 261
  },
  "error": null
}
"""

with open(file_path, "r") as file:
    for line in file:
        audio_url = line.strip()  # Remove any leading/trailing whitespace
        initiateVideoanimation(audio_url, video_url, model, callback_url, False)
