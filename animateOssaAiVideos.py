from transformers import AutoTokenizer, OPTForCausalLM
from diffusers import StableDiffusionPipeline
import subprocess
from pydub import AudioSegment
import requests

import torch
import os
import requests

from glob import glob
import shutil
from time import  strftime
import os, sys, time
from argparse import ArgumentParser
from functools import lru_cache

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from src.utils.publishAnimatedVideo import publishVideo
import inferGen

# Beam Volume to store cached models
cache_path = "./models"
data_files = "./data"
output_files = "./output"

model_id = "runwayml/stable-diffusion-v1-5"
model_script_path = '/workspace/scripts/download_models.sh'


def write_files(output, fileToWrite):
    with open(f"{output}/{fileToWrite}", "w") as f:
        f.write("Writing to the volume!")


def read_files(output, fileToRead):
    with open(f"{output}/{fileToRead}", "r") as f:
        f.read()


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


def generate_image(**inputs):
    start_time = time.time()

    # Grab inputs passed to the API
    try:
        prompt = inputs.get("prompt", "a renaissance style photo of elon musk")
        audio = inputs.get("audio", "./data/audio.wav")  # Specify a real default if applicable
        isImageAnimation = inputs.get("still", True)  # Assuming False as a sensible default
        imageToBeAnimated = inputs.get("image", "./data/jesus.png")  # Specify a real default if applicable
#        result_dir = output_files
        seed_video_path = inputs.get("seed", "XXX")
        character = inputs.get("character", "jesus")

        current_root_path = '/workspace'

        testEnvironment = inputs.get("test", False)  # Assuming False as a sensible default

        #save_dir = os.path.join("result_dir", strftime("%Y_%m_%d_%H.%M.%S"))
        save_dir = "./data/samples"       
        print(f"save_dir {save_dir} ")

        if not os.path.exists(save_dir):
            print(f"Lets create {save_dir} ")
            os.makedirs(save_dir, exist_ok=True)

        print(f"Inputs: audio path {audio} image to be animated {imageToBeAnimated} save Path {save_dir} current_root_path {current_root_path}")

        device = "cuda"
        size = 256
        old_version = False
        preprocess = 'crop'
        checkpoint_dir = 'checkpoints'
        sadtalker_paths = init_path(cache_path, checkpoint_dir, os.path.join(current_root_path, 'src/config'), 256, False, 'crop')
        ref_eyeblink = None
        ref_pose = None
        pose_style = 0
        face3dvis = False
        batch_size = 2
        input_yaw_list = None
        input_pitch_list = None
        input_roll_list = None
        expression_scale = 1.
        enhancer = None
        background_enhancer = None
        verbose = False

        print(f"sadtalker_paths  {sadtalker_paths} ")


    # Use a default prompt if none is provided
    except KeyError:
        prompt = "a renaissance style photo of elon musk"
        print("Handle me - TODO")

    #init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
    
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)

    event_marker = time.time()  # Record the marker time after the code execution
    print(f"3DMM Extraction for source image: {event_marker - start_time} seconds")

    print(f"Image to be animated  {imageToBeAnimated} ")
    print(f"Audio to be lip Synced  {audio} ")

    #TODO - This can be cached
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(imageToBeAnimated, first_frame_dir, preprocess,\
                                                                             source_image_flag=True, pic_size=size)

    print(f"first_coeff_path is {first_coeff_path}, crop_pic_path is {crop_pic_path}, crop_info is {crop_info}")

    event_marker = time.time()  # Record the marker time after the code execution
    print(f"preprocess_model.generate done in: {event_marker - start_time} seconds")
    
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)

        event_marker = time.time()  # Record the marker time after the code execution
        print(f"3DMM Extraction for the reference video providing eye blinking: {event_marker - start_time} seconds")

        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)

            event_marker = time.time()  # Record the marker time after the code execution
            print(f"3DMM Extraction for the reference video providing pose: {event_marker - start_time} seconds")

            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None

    print(f"first_coeff_path is {first_coeff_path}, audio is {audio}, device is {device}, ref_eyeblink_coeff_path is {ref_eyeblink_coeff_path}, isImageAnimation is {isImageAnimation}")

    #audio - mp3 to Wav
    converted_audio_path = convertAudioToWav(audio)
    print("Exported MP3 to WAV path is at ", converted_audio_path)

    event_marker = time.time()  # Record the marker time after the code execution
    print(f"MP3 to WAV conversion done in: {event_marker - start_time} seconds")

    #Set Audio to Downloaded and converted WAV
    audio = converted_audio_path

    #audio2ceoff
    batch = get_data(first_coeff_path, audio, device, ref_eyeblink_coeff_path, isImageAnimation)

    event_marker = time.time()  # Record the marker time after the code execution
    print(f"Audio Data batched in: {event_marker - start_time} seconds")

    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    event_marker = time.time()  # Record the marker time after the code execution
    print(f"coeff_path done in: {event_marker - start_time} seconds")

    # 3dface render
    if face3dvis:
        from src.face3d.visualize import gen_composed_video
        #gen_composed_video(args, device, first_coeff_path, coeff_path, audio, os.path.join(save_dir, '3dface.mp4'))
    
    event_marker = time.time()  # Record the marker time after the code execution
    print(f"Before Face Render: {event_marker - start_time} seconds")

    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=expression_scale, still_mode=isImageAnimation, preprocess=preprocess, size=size)
    
    event_marker = time.time()  # Record the marker time after the code execution
    print(f"After Face Render: {event_marker - start_time} seconds")

    # This is the most expensive step in the Rendering TODO
    result = animate_from_coeff.generate(data, save_dir, imageToBeAnimated, crop_info, seed_video_path, character, \
                                enhancer=enhancer, background_enhancer=background_enhancer, preprocess=preprocess, img_size=size)
    
    event_marker = time.time()  # Record the marker time after the code execution
    print(f"After Generate steps: {event_marker - start_time} seconds")
    print(f"Before publishing: result is {result} and save_dir is {save_dir}")

    videoName = os.path.basename(result)
    publishVideo(result, videoName)
    
    shutil.copy(result, save_dir+'.mp4')
    '''
    filepath = os.path.join(save_path, temp_file)        
        if not os.path.exists(filepath):
            os.makedirs(filepath, exist_ok=True)
    '''

    event_marker = time.time()  # Record the marker time after the code execution
    print(f"The generated video is available: {event_marker - start_time} seconds")

    print('Final Update generated video is named:', save_dir+'.mp4')

    if not verbose:
        print('Handle Me')
        #shutil.rmtree(save_dir)

def convertAudioToWav(mp3Path) :
    print("Export")

    response = requests.get(mp3Path)

    try:
        # Define the path where you want to save the MP3 file
        filepath = './data/samples/downloaded_file.mp3'
        with open(filepath, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved as {filepath}")

        # Load the mp3 file
        audio = AudioSegment.from_mp3(filepath)
        wav_audio_name = os.path.splitext(os.path.split(filepath)[-1])[0]
        print("Exporting file by name", wav_audio_name)

        wav_path = "./data/samples/"+wav_audio_name+".wav"

        # Export as wav
        audio.export(wav_path, format="wav")
        print("Exported to path ", wav_path)
    except Exception as e:
        print("Exception while converting to WAV", e)

    return wav_path
