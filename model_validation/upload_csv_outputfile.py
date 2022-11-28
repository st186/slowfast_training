import re
import boto3
import os
import pandas as pd
from datetime import datetime as d

import constants
from constants import tz, fmt


def upload_output_files_in_s3(s3Path, checkpointFolderName):
    """
    Upload latest Checkpoint to s3 bucket for given s3Path

    Parameters
    ----------
    s3Path : str
        s3 path

    checkpointFolderName: str
        checkpoint folder name

    Returns
    -------
        It upload given checkpoint to s3 bucket

    """
    s3 = boto3.resource('s3',
                        region_name=os.getenv('REGION_NAME'),
                        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))

    bucket_name = s3Path.split("/")[2]
    print(os.listdir(constants.resultOutputPath))
    for output_file in os.listdir(constants.resultOutputPath):
        try:
            s3_upload_path = "/".join(s3Path.split("/")[3:]) + f'val_result/{checkpointFolderName}/{output_file}'
            local_video_path = os.path.join(constants.resultOutputPath, output_file)
            s3.Bucket(bucket_name).upload_file(local_video_path, s3_upload_path)
            print(f"[INFO] {d.now(tz).strftime(fmt)} : {output_file} : Successfully Inserted in S3 bucket : {s3_upload_path}...")

        except Exception as e:
            print(e)
            return