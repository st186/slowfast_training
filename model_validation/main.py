import glob
import imp
import os
import shutil
from datetime import datetime as d

import requests

import model_evaluation
from constants import tz, fmt
import constants
from download_checkpoint import get_checkpoint_from_s3, get_class_name_json_file, get_val_data
import model_execution
import yaml
import json
import torch
from dotenv import load_dotenv
import docker


def get_class_count():
    with open(os.path.join(constants.modelCheckpointBasePath, 'class_name.json'), 'r') as f:
        data = json.load(f)
    
    return len(data)


def remove_docker_container(container_name):
    client = docker.from_env()
    container_list = client.containers.list(all=True)
    for container in container_list:
        if container_name in container.name:
            print(f"[WARNING] {d.now(tz).strftime(fmt)} : Docker Container : {container_name} is found")
            print(f"[WARNING] {d.now(tz).strftime(fmt)} : Docker Container : {container_name} is being removed")
            container.remove(force=True)
            print(f"[WARNING] {d.now(tz).strftime(fmt)} : Docker Container : {container_name} is removed")


def callFinalAPI(result):

    print(f"....................... Loss : {result}")
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    r = requests.post(constants.workstationAPI, data=json.dumps(result), headers=headers)
    print("[info] status_code : ", r.status_code)
    print("[info] response : ", r.json())

    if r.status_code != 204:
        print()
        print(r.json())
        print("\n************ Request is Successfully Sent *****************\n")


if __name__ == "__main__":
    load_dotenv()
    
    # s3Path='s3://production-deployment/test_env/WS1/'
    # checkpoint_dir_name = '2022-01-22 04:05:17'
    os.system('nvidia-smi')
    s3Path = os.environ['s3Path']
    s3Path = s3Path.replace('"', '')

    checkpoint_dir_name = os.environ['checkpoint_dir_name']
    checkpoint_dir_name = checkpoint_dir_name.replace('"', '')

    # Download latest checkpoint
    checkpoint_path = os.path.join(s3Path, 'checkpoints', checkpoint_dir_name)    
    get_checkpoint_from_s3(checkpoint_path)

    # Download class name.json file
    # class_json_path = os.path.join(s3Path, 'label_data')
    get_class_name_json_file(s3Path)

    # Download validation data
    val_data_path = os.path.join(s3Path, 'val_data')
    get_val_data(val_data_path)

    # copy ground truth to ground truth folder
    files = glob.iglob(os.path.join(constants.valInputVideosPath, "*.csv"))
    for file in files:
        if os.path.isfile(file):
            shutil.move(file, constants.valGTPath)

    # get class count
    total_label = get_class_count()

    # prediction
    print(f"[INFO] {d.now(tz).strftime(fmt)} : Model testing is successfully started...")
    os.system(f"""sh /app/model_testing_process.sh \
                     '{total_label}' '{s3Path}' "{checkpoint_dir_name}"
                """)
    print(f"[INFO]{d.now(tz).strftime(fmt)} : Model testing is successfully Completed...")


    # evaluate the model
    result = model_evaluation.main()
    # callFinalAPI(result)

    # remove docker
    # docker_container_name = 'data-validation-' + s3Path.strip("/").split("/")[-1]
    # remove_docker_container(docker_container_name)
    

# docker run -it --rm -e s3Path=s3://production-deployment/test_env/WS1/ data_training:v1
#  docker build -t model_testing:v1 .
# docker run -it --rm -v /home/bhavika/Bhavika_Kanani/TechMahindra/server/TechM-server/data_testing/output:/app/output -e s3Path='s3://production-deployment/test_env/WS1/checkpoints/2022-01-22 04:05:17/' -e rtsp='rtsp://admin:Subham@186@115.187.59.165:554/Streaming/channels/101' model_testing:v1
