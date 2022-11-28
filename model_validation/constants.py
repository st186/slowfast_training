import os
import pytz
import datetime

tz = pytz.timezone('Asia/Kolkata')
fmt = "%Y-%m-%d %H:%M:%S"

SAMPLE_DURATION = 40
CFG = "cfg/C2D_8x8_R50.yaml"

FPS = None
WIDTH, HEIGHT = 256, 256


# ------------------------  Testing Model checkpoint Path ---------------------------------
modelCheckpointBasePath = os.path.dirname(os.path.realpath(__file__)) + f'/checkpoints'
if not os.path.isdir(modelCheckpointBasePath):
    os.makedirs(modelCheckpointBasePath)

# ------------------------  Validation data Path ---------------------------------
valInputVideosPath = os.path.dirname(os.path.realpath(__file__)) + f'/val_data'
if not os.path.isdir(valInputVideosPath):
    os.makedirs(valInputVideosPath)

# ------------------------  Validation data Path ---------------------------------
valGTPath = os.path.dirname(os.path.realpath(__file__)) + f'/gt'
if not os.path.isdir(valGTPath):
    os.makedirs(valGTPath)

# ------------------------  Output data Path ---------------------------------
resultOutputPath = os.path.dirname(os.path.realpath(__file__)) + f'/output_data'
if not os.path.isdir(resultOutputPath):
    os.makedirs(resultOutputPath)

workstationAPI = 'http://10.130.96.89:4001/node/api/v2/cu/getValidationScore'

# # ------------------------  Training Raw Data Path ---------------------------------
# trainingDataBasePath = os.path.dirname(os.path.realpath(__file__)) + f'/data'
# if not os.path.isdir(trainingDataBasePath):
#     os.makedirs(trainingDataBasePath)


# # ------------------------  Training ReScaled Data Path ---------------------------------
# trainingRescaleDataBasePath = os.path.dirname(os.path.realpath(__file__)) + f'/rescale_data'
# if not os.path.isdir(trainingRescaleDataBasePath):
#     os.makedirs(trainingRescaleDataBasePath)


# # ---------------------- FrameList Path ------------------------------
# trainingFrameList= os.path.dirname(os.path.realpath(__file__)) + f'/frame_list'
# if not os.path.isdir(trainingFrameList):
#     os.makedirs(trainingFrameList)

# #  ---------------- Fraction ---------------------------
# trainFraction = 0.8
# valFraction = 0.2


# # ---------------------- Checkpoint Path For Upload ------------------------------
# checkPointBasePathForUpload = os.path.dirname(os.path.realpath(__file__)) + f'/checkpoints'
# # if not os.path.isdir(checkPointBasePathForUpload):
# #     os.makedirs(checkPointBasePathForUpload)