import boto3
import os
from datetime import datetime as d
from constants import tz, fmt
from constants import modelCheckpointBasePath, valInputVideosPath


def get_class_name_json_file(path):
    s3_client = boto3.client('s3',
                             region_name=os.getenv('REGION_NAME'),
                             aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                             aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))

    bucket_name = path.split("/")[2]

    s3_class_name_path = "/".join(path.split('/')[3:])
    s3_class_name_path = os.path.join(s3_class_name_path, 'class_name.json')

    dst_dir_name = os.path.join(modelCheckpointBasePath, 'class_name.json')
    s3_client.download_file(bucket_name, s3_class_name_path, dst_dir_name)

    print(f"[INFO] {d.now(tz).strftime(fmt)} : Successfully download Class_name.json from S3 bucket {s3_class_name_path}")


def get_checkpoint_from_s3(s3CheckpointPath):
    """
    Collect checkpoint from s3 and save to local server

    get_checkpoint_from_s3('s3://production-deployment/test_env/WS1/checkpoints/2022-01-22 04:05:17')

    Parameters
    ----------
    s3Path: str
        Workstation path
    
    Returns
    -------

    """

    s3_client = boto3.client('s3',
                             region_name=os.getenv('REGION_NAME'),
                             aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                             aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))

    bucket_name = s3CheckpointPath.split("/")[2]

    s3_checkpoit_path = "/".join(s3CheckpointPath.split('/')[3:])
    s3_checkpoit_path = os.path.join(s3_checkpoit_path, 'checkpoint_epoch_')

    print('\n s3_checkpoint_path : ', s3_checkpoit_path)

    results = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_checkpoit_path, Delimiter='/')
    contents = results.get('Contents')

    print("\n contents : ", contents)

    get_last_modified = lambda obj: int(obj['LastModified'].strftime('%s'))
    contents = [obj['Key'] for obj in sorted(contents, key=get_last_modified)]

    last_modified_checkpoint = contents[-1]
    checkpoint_name = 'checkpoint.pyth'

    if '.pyth' not in last_modified_checkpoint:
        return f"checkpoint is not exist at {last_modified_checkpoint}"

    dst_dir_name = os.path.join(modelCheckpointBasePath, checkpoint_name)
    s3_client.download_file(bucket_name, last_modified_checkpoint, dst_dir_name)

    print(f"[INFO] {d.now(tz).strftime(fmt)} : Successfully download model checkpoint from S3 bucket {last_modified_checkpoint}...")


def get_val_data(path):
    s3 = boto3.resource('s3',
                        region_name=os.getenv('REGION_NAME'),
                        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))

    bucket_name = path.split("/")[2]
    class_name_path = "/".join(path.split('/')[3:])

    bucket = s3.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=class_name_path):
        target = obj.key if valInputVideosPath is None \
            else os.path.join(valInputVideosPath, os.path.relpath(obj.key, class_name_path))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
