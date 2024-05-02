import shutil
import subprocess
import uuid

import os

import cv2

def load_video_to_cv2(input_path):
    video_stream = cv2.VideoCapture(input_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = [] 
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break 
        full_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return full_frames

def download_file(url, directory, filename):
    """
    Download a file from a given URL to a specified directory.
    """
    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Complete file path
    filepath = os.path.join(directory, filename)

    # Check if file already exists to avoid re-downloading
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {filename} successfully at {filepath}.")
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")
    else:
        print(f"{filename} already exists. Skipping download.")

def save_video_with_watermark(video, audio, save_path, character, watermark=False):
    temp_file = str(uuid.uuid4())+'_'+character+'.mp4'

    save_path = "./data/samples"
#    temp_file = f"{character}.mp4"
    temp_file_path = os.path.join(save_path, temp_file)

    print(f'Before ffmeg video {video} and audio {audio} and save_path is {save_path} and temp_file {temp_file} and temp_file_path is {temp_file_path}') 

    # Use subprocess to run ffmpeg
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'info',
        '-i', video, '-i', audio, '-vcodec', 'copy', temp_file_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("FFmpeg error:", e.stderr.decode())
 
    #save_path = "./data/samples"
    returnPath = temp_file_path

    print(f'After ffmeg The generated video is named {temp_file} and needs to go to save_path {save_path}')

    print(f'After ffmeg returned path is {returnPath}')

    if watermark is False:
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        '''
        '''
        try:
            shutil.move(temp_file, save_path)
        except Exception as e:
            print("Exception is", e)
        '''
    else:
        # watermark
        try:
            ##### check if stable-diffusion-webui
            import webui
            from modules import paths
            watarmark_path = paths.script_path+"/extensions/SadTalker/docs/sadtalker_logo.png"
        except:
            # get the root path of sadtalker.
            dir_path = os.path.dirname(os.path.realpath(__file__))
            watarmark_path = dir_path+"/../../docs/sadtalker_logo.png"

        cmd = r'ffmpeg -y -hide_banner -loglevel error -i "%s" -i "%s" -filter_complex "[1]scale=100:-1[wm];[0][wm]overlay=(main_w-overlay_w)-10:10" "%s"' % (temp_file, watarmark_path, save_path)
        os.system(cmd)
        #os.remove(temp_file)

    return returnPath