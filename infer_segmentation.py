# Script to Perform Inference on the Trained Model of Segmentation Problem

import os
import argparse
import torch
import sys
import webcolors
import numpy as np
from pointnet.model import PointNetDenseCls
import open3d as o3d

parser = argparse.ArgumentParser()

parser.add_argument("--cad_folder", required=True, help="Folder containing 3D CAD Models for prediction")
parser.add_argument("--trained_nn_model", required=True, help="Trained Neural Network Model")
parser.add_argument("--output_folder", required=True, help="Path to Output Folder")

args = parser.parse_args()

cad_model_folder = args.cad_folder
saved_model = args.trained_nn_model
output_folder = args.output_folder

# few hardcoded stuff 
feature_transform = None

#input check 
if not os.path.isdir(cad_model_folder):
    print(f"{cad_model_folder} : not a directory")
    sys.exit()

os.makedirs(output_folder, exist_ok=True)
    
######################################
# UTILITY FUNCTION
colorMap = {0: "red", 1: "green", 2: "orange", 3: "cyan"}
def get_color(id):
    color_array = np.asarray(webcolors.name_to_rgb(colorMap[id]))
    return color_array
#######################################

# Load the Model from the saved path
device = torch.device('cpu') # if Running on CPU
state_dict = torch.load(saved_model, map_location=device)

model = PointNetDenseCls(k = state_dict['conv4.weight'].size()[0])
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
        # choice = np.random.choice(len(pointSet), num_points, replace=True)
        # pointSet = pointSet[choice, :]

        # Normalize and center and bring it to unit sphere
        pointSet = pointSet - np.expand_dims(np.mean(pointSet, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(pointSet ** 2, axis = 1)),0)
        pointSet = pointSet / dist #scale
        # THE ABOVE STEPS ARE NOT DONE BY fxia22 IN HIS REPOSITORY 

        # convert to pytorch tensor
        points = torch.from_numpy(pointSet)
        points = points.float() #model requires float32
        points = torch.unsqueeze(points, 0)
        points = points.transpose(2, 1)
        #points = points.cuda() #uncomment this if running on GPU
        model = model.eval()
        pred, _, _ = model(points)

        pred_choice = pred.data.max(2)[1]
        pred_choice = torch.squeeze(pred_choice)
        print(f"pred_choice.shape: {pred_choice.shape}")
        print(f"{pred_choice}")

        color_matrix = np.empty(pointSet.shape)
        for i in range(color_matrix.shape[0]):
            seg_id = pred_choice[i].item()
            color_matrix[i] = get_color(seg_id)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointSet)
        pcd.colors = o3d.utility.Vector3dVector(color_matrix)
        outPcdFile = os.path.join(output_folder, os.path.splitext(fileName)[0] + "_out.pcd")
        o3d.io.write_point_cloud(outPcdFile, pcd, write_ascii=True)



