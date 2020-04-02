# create a two text file with train and test shuffles

# output will be of the following format 
# {
#   train: {
#       file1:label-A
#       file2:label-B
#       ...
#   }
#   test: {
#       file1:label-A
#       file2:label-B
#       ...
#   }
# }

# inputs
# 1. dataset path 
# it is assumed that the path will contain labled folders
#   label-A
#       file1
#       file2
#   label-B
#       file1
#       file2
# 2. split percent

import os
import random
import math
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_path", required=True, help="Path to the Dataset Folder")
parser.add_argument("--split_percent", default=80, help="Split Percentage between Train and Test")

args = parser.parse_args()

# datasetPath = r"F:\projects\ai\pointnet\dataset\DMUNet_OBJ_format\dataset_PCD_5000"
# splitPercent = 80

datasetPath = args.input_path
splitPercent = args.split_percent

outputFilePath = os.path.join(datasetPath, "train_test_split.json")

train_test_Dict = {"train": {}, "test": {}}

for labelDir in os.listdir(datasetPath):
    labelDirPath = os.path.join(datasetPath, labelDir)
    if os.path.isdir(labelDirPath):
        files = os.listdir(labelDirPath)

        random.shuffle(files)
        files = [os.path.join(datasetPath, labelDir, f) for f in files]
        
        numFiles = len(files)
        splitIndex = math.floor(numFiles * (splitPercent/100.0))

        trainFiles = files[:splitIndex]
        testFiles = files[splitIndex:]

        train_test_Dict["train"].update({f:labelDir for f in trainFiles})
        train_test_Dict["test"].update({f:labelDir for f in testFiles})

        print(f"debug: {labelDir} => totalFiles: {numFiles}, train: {len(trainFiles)} + test: {len(testFiles)} = {len(trainFiles) + len(testFiles)} ")

with open(outputFilePath, 'w') as outFile:    
    json.dump(train_test_Dict, outFile, indent=4)
    







