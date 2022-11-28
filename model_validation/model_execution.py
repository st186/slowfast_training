import datetime
import json
import os
import signal
import sys
from datetime import datetime as d
from constants import tz, fmt
import numpy as np
import time

import pandas as pd
import requests
import torch
import tqdm
import constants
from upload_csv_outputfile import upload_output_files_in_s3
from slowfast.utils.parser import parse_args, load_config
from slowfast.utils import logging
from slowfast.visualization.async_predictor import AsyncDemo, AsyncVis

from video_decoder.decode import VideoManager
from slowfast.visualization.predictor import ActionPredictor
from slowfast.visualization.video_visualizer import VideoVisualizer

logger = logging.get_logger(__name__)

prev_pred = []


def extract_prediction(preds):
    global prev_pred
    pred_result = {}
    top_class = int(np.argmax(preds).tolist())

    if top_class in prev_pred:
        top_class = prev_pred[-1]
    else:
        prev_pred.append(top_class)
    pred_result['description'] = pred_classes[top_class]
    pred_result['score'] = preds.tolist()[0][top_class]
    pred_result['seqNo'] = int(top_class) + 1

    return pred_result


def postprocessing(data):
    df = pd.DataFrame(data).T
    df = df.rename(columns={'description': 'action'})

    df['action'] = df['action'].fillna('NA')
    last_label = None
    idx = None

    for i in range(len(df)):
        if df.loc[i, 'action'] == last_label:
            df.loc[idx, 'end_time'] = df.loc[i, 'end_time']
            df = df.drop(i, axis=0)
        else:
            last_label = df.loc[i, 'action']
            idx = i

    df['action'] = df['action'].replace('NA', np.nan)
    return df


def run_demo(cfg, frame_provider):
    """
    Run demo visualization.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast2/config/defaults.py
        frame_provider (iterator): Python iterator that return task objects that are filled
            with necessary information such as `frames`, `id` and `num_buffer_frames` for the
            prediction and visualization pipeline.
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    # Print config.
    # logger.info("Run demo with config:")
    # logger.info(cfg)
    common_classes = (
        cfg.DEMO.COMMON_CLASS_NAMES
        if len(cfg.DEMO.LABEL_FILE_PATH) != 0
        else None
    )
    print(f"......................classes : {cfg.MODEL.NUM_CLASSES}")
    # all model configuration such as threshold_val : 0.7, lower_threshold: 0.3 and etc
    video_vis = VideoVisualizer(
        num_classes=cfg.MODEL.NUM_CLASSES,
        class_names_path=cfg.DEMO.LABEL_FILE_PATH,
        top_k=cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        thres=cfg.DEMO.COMMON_CLASS_THRES,
        lower_thres=cfg.DEMO.UNCOMMON_CLASS_THRES,
        common_class_names=common_classes,
        colormap=cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
        mode=cfg.DEMO.VIS_MODE,
    )

    async_vis = AsyncVis(video_vis, n_workers=cfg.DEMO.NUM_VIS_INSTANCES)

    if cfg.NUM_GPUS <= 1:
        model = ActionPredictor(cfg=cfg, async_vis=async_vis)
    else:
        model = AsyncDemo(cfg=cfg, async_vis=async_vis)

    seq_len = constants.SAMPLE_DURATION

    assert (
            cfg.DEMO.BUFFER_SIZE <= seq_len // 2
    ), "Buffer size cannot be greater than half of sequence length."
    num_task = 0
    # Start reading frames.
    frame_provider.start()
    for able_to_read, task in frame_provider:
        if not able_to_read:
            break
        if task is None:
            time.sleep(0.02)
            continue
        num_task += 1

        model.put(task)
        try:
            task = model.get()
            num_task -= 1
            yield task
        except IndexError:
            continue

    while num_task != 0:
        try:
            task = model.get()
            num_task -= 1
            yield task
        except IndexError:
            continue


def run(cfg, ip_video_path, op_video_path, roi_dix):
    """
    Run inference on an input video or stream from webcam.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast2/config/defaults.py
        ip_video_path (str): input video file path
        op_video_path (str): output video file path

    Returns:
        Dict[dict]:
            dict of prediction for example : dict[dict] : {0:['top_code':0.99, 'top_class':'Start of Assembly']

    """
    # AVA format-specific visualization with precomputed boxes.

    start = time.time()

    frame_provider = VideoManager(cfg, ip_video_path, op_video_path, roi_dix)

    pred_result = []
    for task in tqdm.tqdm(run_demo(cfg, frame_provider)):
        result = extract_prediction(task.action_preds)
        pred_data = {'start_time': task.start_time, 'end_time': task.end_time}
        pred_data.update(result)
        pred_result.append(pred_data)
        print("\n**************************")
        print(f"Summary :"
              f"\n {result}"
              f"\n Start Time : {task.start_time}"
              f"\n End Time : {task.end_time}"
              f"\n Total Frames : {task.frame_list_len}")
        print("**************************\n")
        frame_provider.display(task)

        if time.time() - start > 50:
            break

    frame_provider.join()
    frame_provider.clean()
    logger.info("Finish demo in: {}".format(time.time() - start))

    return dict(enumerate(pred_result))


if __name__ == "__main__":

    roi_dix = {'x': 290, 'y': 50, 'w': 1160, 'h': 720}

    # parse arguments
    args = parse_args()
    cfg = load_config(args)
    checkpointFolderName = args.checkpoint_folder_name
    s3Path = args.s3Path

    print(f"[INFO] {d.now(tz).strftime(fmt)} : S3 Path : {s3Path} || CheckPoint Folder Path : {checkpointFolderName}")

    # prediction classes
    pred_classes = json.load(open(os.path.join(constants.modelCheckpointBasePath, 'class_name.json')))
    pred_classes = {value: key for key, value in pred_classes.items()}
    print("[info] pred_classes : ", pred_classes)

    # work on all videos
    for video_name in os.listdir(constants.valInputVideosPath):
        # output video
        print(
            f"[INFO] {d.now(tz).strftime(fmt)} : -------------------- Input Video File Name : {video_name} -------------------- ")
        ip_video_path = os.path.join(constants.valInputVideosPath, video_name)
        op_video_path = os.path.join(constants.resultOutputPath, video_name)
        pred_result = run(cfg, ip_video_path, op_video_path, roi_dix)

        # csv file
        df = postprocessing(pred_result)
        csv_path = f"{constants.resultOutputPath}/{video_name.replace('.mp4', '')}.csv"
        df.to_csv(csv_path, index=False)

    # save output files in s3 bucket
    upload_output_files_in_s3(s3Path, checkpointFolderName)
    print("Done...............")
    os.kill(os.getpid(),signal.SIGKILL)

