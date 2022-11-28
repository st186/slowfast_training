from numpy import clip
import cv2
import os
import imgaug as ia
from imgaug import augmenters as iaa
import shutil
import random
import time
from vidaug import augmentors as va
from multiprocessing import Pool
import concurrent.futures
import subprocess

ia.seed(1) 

def speed_modifier(i, video, output):

    output_path = os.path.splitext(output)[0]

    if main.upsample == True:
        print(main.upsampling_rate)
        modified_output_path = output_path + "_" + str(i) + "_upsampled" + ".mp4"
        command = "ffmpeg -i {video} -filter:v 'setpts={upsampling_rate}*PTS' -c:v mpeg4 -q:v 2 -an {output}".format(video=video, upsampling_rate=main.upsampling_rate, output=modified_output_path)
        subprocess.call(command,shell=True)
    if main.downsample == True:
        print(main.downsampling_rate)
        modified_output_path = output_path + "_" + str(i) + "_downsampled" + ".mp4"
        command = "ffmpeg -i {video} -filter:v 'setpts={downsampling_rate}*PTS' -c:v mpeg4 -q:v 2 -an {output}".format(video=video, downsampling_rate=main.downsampling_rate, output=modified_output_path)
        subprocess.call(command,shell=True)

def augment_and_save_frames(video_reader, output_folder_path,video_clip_name,i,fps,w,h):

    """
        Fetch each frame of video and augment and save as picture in a temporary folder
        Args:
            video_reader: Video reader object
            rotation_angle: int (Angle of rotation of image)
            noise_value: int (noise value between 0 to 100)
            temp_folder_path: string (temporary path to store video frames)
            output_folder_path: string (output folder path)
            video_clip_name: string (video name)
            i: no of clip augmented
    """
    # These 4 lines take care of abnormal file names
    temp = video_clip_name.replace(" ","")
    temp = temp.split(".")
    editted_name = temp[0]+"_"+str(i)+"."+temp[1]
    path_of_video_to_save = output_folder_path+"//"+editted_name

    seed = i

    sigma = random.uniform(main.low_sigma_range, main.high_sigma_range)
    alpha = random.uniform(main.low_alpha_range, main.high_alpha_range)
    scale = random.uniform(main.low_scale_range, main.high_scale_range)
    mul = random.uniform(main.low_mul_range, main.high_mul_range)

    print("Sigma", sigma)

    seq = iaa.Sequential([
                # iaa.GaussianBlur(sigma=sigma, seed=seed),        
                iaa.ContrastNormalization(alpha=alpha, seed=seed),         
                # iaa.AdditiveGaussianNoise(
                #     loc=0, scale=scale, per_channel=0.5, seed=seed),    
                iaa.Multiply(mul=mul, per_channel=0.2, seed=seed)
    ], random_order= False)

    fourcc = 'mp4v'  # output video codec
    video_writer = cv2.VideoWriter(path_of_video_to_save, cv2.VideoWriter_fourcc(*fourcc),fps,(w,h))

    try:
        while video_reader.isOpened():
            ret, frame = video_reader.read()
            if not ret:
                break
            image_aug = seq(image=frame)
            video_writer.write(image_aug)
    except Exception as e:
        print(e)

    cv2.destroyAllWindows()
    video_reader.release()
    video_writer.release()

def augment_videos(i):

    try:
        video_path = f"{main.main_folder_path}//{video_clip_names[main.clip_no]}"
        print(video_path)
        output_path = f"{main.output_folder_path}//{video_clip_names[main.clip_no]}"
        video_reader = cv2.VideoCapture(video_path)
        fps = int(video_reader.get(cv2.cv2.CAP_PROP_FPS))
        w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Get fps for input video
        print(f"FPS of {video_clip_names[main.clip_no]} is {fps}")  
        start = time.time()
        if main.is_aug:
            augment_and_save_frames(video_reader,main.output_folder_path,video_clip_names[main.clip_no],i,fps,w,h)
        if main.is_temporal_aug:
            speed_modifier(i, video_path, output_path)
        end = time.time()
        print("Total time taken by single video", end-start)
    except Exception as e:
        print("Exception is", e)

time_of_code = time.time()

def main(main_folder_path, output_folder_path, no_of_clips_to_augment_per_frame, low_sigma_range, high_sigma_range, low_alpha_range, high_alpha_range, low_scale_range, high_scale_range, low_mul_range, high_mul_range, is_aug, is_temporal_aug, downsample, upsample, downsampling_rate, upsampling_rate):

    main.low_sigma_range=low_sigma_range
    main.high_sigma_range=high_sigma_range
    main.low_alpha_range=low_alpha_range
    main.high_alpha_range=high_alpha_range
    main.low_scale_range=low_scale_range
    main.high_scale_range=high_scale_range
    main.low_mul_range=low_mul_range
    main.high_mul_range=high_mul_range
    main.is_aug = is_aug
    main.is_temporal_aug = is_temporal_aug
    main.main_folder_path = main_folder_path
    main.output_folder_path = output_folder_path   

    main.upsample = upsample
    main.downsample = downsample
    main.upsampling_rate = upsampling_rate
    main.downsampling_rate = downsampling_rate 

    print("Output folder path", output_folder_path)
    print("Main folder path", main_folder_path)
    print("Max augmented clips", no_of_clips_to_augment_per_frame)
        
    if os.path.exists(output_folder_path) and os.path.isdir(output_folder_path):
        shutil.rmtree(output_folder_path)
    os.makedirs(output_folder_path,exist_ok=True)
    global video_clip_names
    global no_of_clips_available
    main.clip_no = 0
    
    video_clip_names = os.listdir(main_folder_path)
    print(f"Videos found are {video_clip_names}")
    no_of_clips_available = len(video_clip_names)

    # Run for each clip that needs to be augmented
    for main.clip_no in range(no_of_clips_available):
        # Rotate the clip based on angle range and increment the subsequent clips w.r.t. the angle increment
        print("No. of videos to be augmented per input", no_of_clips_to_augment_per_frame)
        with concurrent.futures.ThreadPoolExecutor() as executor:       
            print(list(range(no_of_clips_to_augment_per_frame)))        
            executor.map(augment_videos, list(range(no_of_clips_to_augment_per_frame)))

# main(main_folder_path = 'sample_videos/', output_folder_path='augmented_videos/', no_of_clips_to_augment_per_frame = 3, low_sigma_range=0.0, high_sigma_range=0.5, low_alpha_range=0.75, high_alpha_range=1.5, low_scale_range=0.0, high_scale_range=0.05 * 255, low_mul_range=0.8, high_mul_range=1.2, is_aug=True, is_temporal_aug=True, downsample=True, upsample=True, downsampling_rate=1.25, upsampling_rate=0.75)


# -----------------------------------------------------------------------

def create_augmented_videos(
        raw_video_folder_path: str,
        augmented_video_folder_path: str,
):
    """
    Create augmented videos from give local directory video path
    Parameters
    ----------
    raw_video_folder_path: str
        where the raw videos is saved
    augmented_video_folder_path: str
        where the augmented videos are going to be saved
    data_training_var:dict

    Returns
    -------

    """
    for sub_dir in os.listdir(raw_video_folder_path):
        print("[info] sub_dir : ", sub_dir)

        # action1, action2, ...
        sub_raw_video_dir_full_path = os.path.join(raw_video_folder_path, sub_dir)
        print("sub_raw_video_dir_full_path : ", sub_raw_video_dir_full_path)


        main_folder_path = sub_raw_video_dir_full_path
        output_folder_path = f"{augmented_video_folder_path}/{sub_dir}"

        print("[Aug] input folder name : ", main_folder_path)
        print("[Aug] output folder name : ", output_folder_path)

        print("before augmentation : ", len(os.listdir(main_folder_path)))

        main(main_folder_path = main_folder_path, output_folder_path=output_folder_path, no_of_clips_to_augment_per_frame = 3, low_sigma_range=0.0, high_sigma_range=0.5, low_alpha_range=0.75, high_alpha_range=1.5, low_scale_range=0.0, high_scale_range=0.05 * 255, low_mul_range=0.8, high_mul_range=1.2, is_aug=True, is_temporal_aug=True, downsample=True, upsample=True, downsampling_rate=1.25, upsampling_rate=0.75)

        print("After augmentation : ", len(os.listdir(output_folder_path)))



# ------------------------------------------------------------------------------

create_augmented_videos(raw_video_folder_path='/home/softmaxai/Bhavika_Kanani/TechMahindra/TechM-Training-2904/rescaled_data',
                        augmented_video_folder_path = '/home/softmaxai/Bhavika_Kanani/TechMahindra/TechM-Training-2904/aug_data')