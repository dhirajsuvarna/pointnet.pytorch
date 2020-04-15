# Script to Perform Inference on the Trained Model for Classification Problem
#
import os
import sys
import argparse
import torch
import numpy as np
from pointnet.model import PointNetCls, feature_transform_regularizer
import open3d as o3d

# parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("--cad_folder", required=True, help="Folder containing 3D CAD Models for prediction")
parser.add_argument("--trained_nn_model", required=True, help="Trained Neural Network Model")
parser.add_argument("--class_lables_path", required=True, help="Path to the file containing class Labels")

args = parser.parse_args()

classLabelsPath = args.class_lables_path
saved_model = args.trained_nn_model
cad_model_folder = args.cad_folder

# few hardcoded stuff 
num_points = 4000
feature_transform = None

#input check 
if not os.path.isdir(cad_model_folder):
    print(f"{cad_model_folder} : not a directory")
    sys.exit()

# Read ClassNames and store in List
with open(classLabelsPath, 'r') as classFile:
    classLabels = classFile.read().splitlines()

num_classes = len(classLabels)

# Load the Model from the saved path
# if Running on CPU
device = torch.device('cpu')
state_dict = torch.load(saved_model, map_location=device)

# if Running on GPU
# state_dict = torch.load(saved_model) #uncomment this if running on GPU
model = PointNetCls(k=num_classes, feature_transform=feature_transform)
model.load_state_dict(state_dict)


for root, subdirs, files in os.walk(cad_model_folder):
    for fileName in files:
        cadFilePath = os.path.join(root, fileName)
        # check the file 
        if not cadFilePath.endswith('.pcd'):
            print(f"{cadFilePath}: not a valid pcd file")
            continue

        # read point cloud data
        cloud = o3d.io.read_point_cloud(cadFilePath)
        pointSet = cloud.points

        # THE BELOW STEPS ARE NOT DONE BY fxia22 IN HIS REPOSITORY 
        # extract only "N" number of point from the Point Cloud
        choice = np.random.choice(len(pointSet), num_points, replace=True)
        pointSet = pointSet[choice, :]

        # Normalize and center and bring it to unit sphere
        pointSet = pointSet - np.expand_dims(np.mean(pointSet, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(pointSet ** 2, axis = 1)),0)
        pointSet = pointSet / dist #scale
        # THE ABOVE STEPS ARE NOT DONE BY fxia22 IN HIS REPOSITORY 

        # convert to pytorch tensor
        points = torch.from_numpy(pointSet)
        points = torch.unsqueeze(points, 0)
        points = points.transpose(2, 1)
        #points = points.cuda() #uncomment this if running on GPU
        model = model.eval()
        pred, _, _ = model(points)

        classID = torch.argmax(pred, dim=1)
        className = classLabels[classID]
        print(f"{cadFilePath}: {classID} : {className}")
