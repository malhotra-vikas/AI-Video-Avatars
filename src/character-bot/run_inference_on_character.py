import subprocess

from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

# Global Variables Initialization
global_vars = {
    "global_pic_path":"",
    "global_audio_path":"",
    "global_save_dir":"",
    "global_pose_style":"",
    "global_device":"",
    "global_batch_size":"",
    "global_input_yaw_list":"",
    "global_input_pitch_list":"",
    "global_input_roll_list":"",
    "global_ref_eyeblink":"",
    "global_ref_pose":"",
    "global_pre_process":"",
    "global_arg_size":"",
    "global_args_still":"",
    "global_args_face3d":"",
    "global_args":"",
    "global_crop_pic_path":"",
    "global_crop_info":"",

    "global_current_root_path":"",
    "global_sadtalker_paths":"",

    "global_preprocess_model":"",
    "global_audio_to_coeff":"",
    "global_animate_from_coeff":""
}

# Intermediatories Initialization

intermediate_data = {
    'first_coeff_path': None,
    'ref_eyeblink_coeff_path': None,
    'ref_pose_coeff_path': None,
    'data_for_AVRender': None,  # Placeholder for data generated in ONE_AVtoCoeff
}

def set_global_var(var_name, value):
    global_vars[var_name] = value

def get_global_var(var_name):
    return global_vars[var_name]

def set_intermediatory_var(var_name, value):
    intermediate_data[var_name] = value

def get_intermediatory_var(var_name):
    return intermediate_data[var_name]


def extractPaths(args):
    set_global_var("global_args", args)
    set_global_var("global_pic_path", args.source_image)
    set_global_var("global_audio_path",args.driven_audio)
    set_global_var("global_save_dir", os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S")))

    os.makedirs(get_global_var("global_save_dir"), exist_ok=True)
    set_global_var("global_pose_style", args.pose_style)
    set_global_var("global_device", args.device)
    set_global_var("global_batch_size", args.batch_size)
    set_global_var("global_input_yaw_list", args.input_yaw)
    set_global_var("global_input_pitch_list", args.input_pitch)
    set_global_var("global_input_roll_list", args.input_roll)
    set_global_var("global_ref_eyeblink", args.ref_eyeblink)
    set_global_var("global_ref_pose", args.ref_pose)
    set_global_var("global_pre_process", args.preprocess)
    set_global_var("global_arg_size", args.size)
    set_global_var("global_args_still", args.still)
    set_global_var("global_args_face3d", args.face3dvis)

    set_global_var("global_current_root_path", os.path.split(sys.argv[0])[0])

    set_global_var("global_sadtalker_paths", init_path(args.checkpoint_dir, os.path.join(get_global_var("global_current_root_path"), 'src/config'), get_global_var("global_arg_size"), args.old_version, get_global_var("global_pre_process")))

def ZERO_startProcess(parser):
    start_time = time.time()  # Record the start time

    #parser = ArgumentParser()
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
    parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" ) 
    parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 

    print(f"args.face3dvis: {get_global_var('global_args_face3d')} ")


    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')


    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    extractPaths(args)

    initializeModel()

    first_coeff_path, ref_eyeblink_coeff_path, ref_pose_coeff_path = extract3DimmFromImage(get_global_var("global_preprocess_model"))

    event_marker = time.time()  # Record the marker time after the code execution
    print(f"startProcess took: {event_marker - start_time} seconds")

    set_intermediatory_var("first_coeff_path", first_coeff_path)
    set_intermediatory_var("ref_eyeblink_coeff_path", ref_eyeblink_coeff_path)
    set_intermediatory_var("ref_pose_coeff_path", first_coeff_path)


    #return first_coeff_path, ref_eyeblink_coeff_path, ref_pose_coeff_path


def ONE_AVtoCoeff():
    start_time = time.time()  # Record the start timer

    coeff_path = audio2Coeff(get_intermediatory_var("first_coeff_path"), get_intermediatory_var("ref_eyeblink_coeff_path"), get_intermediatory_var("ref_pose_coeff_path"))

    data = video2Coeff(coeff_path, get_intermediatory_var("first_coeff_path"))

    event_marker = time.time()  # Record the marker time after the code execution
    print(f"AVtoCoeff took: {event_marker - start_time} seconds")

    set_intermediatory_var("data_for_AVRender", data)

    #return data

def TWO_AVRender():
    start_time = time.time()  # Record the start timer

    generateAudioOverlay(get_intermediatory_var("data_for_AVRender"))

    event_marker = time.time()  # Record the marker time after the code execution
    print(f"AVRender took: {event_marker - start_time} seconds")



def initializeModel():
    #init model
    preprocess_model = CropAndExtract(get_global_var("global_sadtalker_paths"), get_global_var("global_device"))

    audio_to_coeff = Audio2Coeff(get_global_var("global_sadtalker_paths"), get_global_var("global_device"))
    
    animate_from_coeff = AnimateFromCoeff(get_global_var("global_sadtalker_paths"), get_global_var("global_device"))

    # Use set_global_var to update the global_vars dictionary
    set_global_var("global_preprocess_model", preprocess_model)
    set_global_var("global_audio_to_coeff", audio_to_coeff)
    set_global_var("global_animate_from_coeff", animate_from_coeff)


def extract3DimmFromImage(global_preprocess_model):
    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(get_global_var("global_save_dir"), 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)

    first_coeff_path, global_crop_pic_path, global_crop_info =  global_preprocess_model.generate(get_global_var("global_pic_path"), first_frame_dir, get_global_var("global_pre_process"),\
                                                                             source_image_flag=True, pic_size=get_global_var("global_arg_size"))
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if get_global_var("global_ref_eyeblink") is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(get_global_var("global_ref_eyeblink"))[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(get_global_var("global_save_dir"), ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)


        ref_eyeblink_coeff_path, _, _ =  global_preprocess_model.generate(get_global_var("global_ref_eyeblink"), ref_eyeblink_frame_dir, get_global_var("global_pre_process"), source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if get_global_var("global_ref_pose") is not None:
        if get_global_var("global_ref_pose") == get_global_var("global_ref_eyeblink"): 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(get_global_var("global_ref_pose"))[-1])[0]
            ref_pose_frame_dir = os.path.join(get_global_var("global_save_dir"), ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)

            ref_pose_coeff_path, _, _ =  global_preprocess_model.generate(get_global_var("global_ref_pose"), ref_pose_frame_dir, get_global_var("global_pre_process"), source_image_flag=False)
    else:
        ref_pose_coeff_path=None

    return first_coeff_path, ref_eyeblink_coeff_path, ref_pose_coeff_path


#audio2ceoff. We will run this for each audio file that needs to be added to the Video
def audio2Coeff(first_coeff_path, ref_eyeblink_coeff_path, ref_pose_coeff_path):
    #audio2ceoff
    batch = get_data(first_coeff_path, get_global_var("global_audio_path"), get_global_var("global_device"), ref_eyeblink_coeff_path, still=get_global_var("global_args_still"))

    coeff_path = get_global_var("global_audio_to_coeff").generate(batch, get_global_var("global_save_dir"), get_global_var("global_pose_style"), ref_pose_coeff_path)

    print(f"args.face3dvis: {get_global_var('global_args_face3d')} ")

    # 3dface render
    if get_global_var("global_args_face3d"):
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(get_global_var("global_args"), get_global_var("global_device"), first_coeff_path, coeff_path, get_global_var("global_audio_path"), os.path.join(get_global_var("global_save_dir"), '3dface.mp4'))
    
    return coeff_path

#coeff2video. We will run this for each audio file that needs to be added to the Video
def video2Coeff(coeff_path, first_coeff_path):
    data = get_facerender_data(coeff_path, get_global_var("global_crop_pic_path"), first_coeff_path, get_global_var("global_audio_path"), 
                                get_global_var("global_batch_size"), get_global_var("global_input_yaw_list"), get_global_var("global_input_pitch_list"), get_global_var("global_input_roll_list"),
                                expression_scale=get_global_var("global_args").expression_scale, still_mode=get_global_var("global_args_still"), preprocess=get_global_var("global_pre_process"), size=get_global_var("global_arg_size"))
    
    return data

# This is the most expensive step in the Rendering TODO
def generateAudioOverlay(data):
    result = get_global_var("global_animate_from_coeff").generate(data, get_global_var("global_save_dir"), get_global_var("global_pic_path"), get_global_var("global_crop_info"), \
                                enhancer=get_global_var("global_args").enhancer, background_enhancer=get_global_var("global_args").background_enhancer, preprocess=get_global_var("global_pre_process"), img_size=get_global_var("global_arg_size"))
    
    
    shutil.move(result, get_global_var("global_save_dir")+'.mp4')

    print('The generated video is named:', get_global_var("global_save_dir")+'.mp4')

    if not get_global_var("global_args").verbose:
        shutil.rmtree(get_global_var("global_save_dir"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--step", choices=['initilize', 'stage', 'render'], required=True, help="Specify which step to execute.")
    args = parser.parse_args()

    if args.step == 'initilize':
        ZERO_startProcess(parser)
    elif args.step == 'stage':
        ONE_AVtoCoeff()
    elif args.step == 'render':
        TWO_AVRender()


def runCharacterAnimation(driven_audio, source_image, result_dir, still="still"):
    """
    Executes an audio-driven animation command with the given parameters.

    Parameters:
    - driven_audio: Path to the audio file to drive the animation.
    - source_image: Path to the source image to animate.
    - result_dir: Directory to store the resulting animation.
    - still: Optional; specify "still" to run in still mode. Default is "still".
    """
    command = [
        "python3.8",
        "inference.py",
        "--driven_audio", driven_audio,
        "--source_image", source_image,
        "--result_dir", result_dir,
        "--still", still
    ]
    
    try:
        subprocess.run(command, check=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

