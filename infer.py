# Script to Perform Inference on the Trained Model
#
import torch
import numpy as np
from pointnet.model import PointNetCls, feature_transform_regularizer
from pypcd import pypcd

# Load the Model from the saved path
num_classes = 32
feature_transform = None
saved_model = r"F:\projects\ai\pointnet\dhiraj\pointnet.pytorch\saved_models\cls_model_49.pth"
num_points = 4000
classLabelsPath = r"F:\projects\ai\pointnet\dataset\DMUNet_OBJ_format\dataset_PCD_5000\classlabels.txt"


# Read ClassNames and store in List
with open(classLabelsPath, 'r') as classFile:
    classLabels = classFile.read().splitlines()

model = PointNetCls(k=num_classes, feature_transform=feature_transform)

device = torch.device('cpu')
state_dict = torch.load(saved_model, map_location=device)
model.load_state_dict(state_dict)


# write a function to do single inference
def infer_single_model(model, inputModelPath):
    # read point cloud data
    cloud = pypcd.PointCloud.from_path(inputModelPath)
    # convert the structured numpy array to a ndarray
    pointSet = cloud.pc_data.view(np.float32).reshape(cloud.pc_data.shape + (-1,))

    # extract only "N" number of point from the Point Cloud
    choice = np.random.choice(len(pointSet), num_points, replace=True)
    pointSet = pointSet[choice, :]

    # Normalize and center and bring it to unit sphere
    pointSet = pointSet - np.expand_dims(np.mean(pointSet, axis = 0), 0) # center
    dist = np.max(np.sqrt(np.sum(pointSet ** 2, axis = 1)),0)
    pointSet = pointSet / dist #scale

    # convert to pytorch tensor
    points = torch.from_numpy(pointSet)
    points = torch.unsqueeze(points, 0)
    points = points.transpose(2, 1)
    #points = points.cuda()
    model = model.eval()
    pred, _, _ = model(points)
    return pred


model_to_infer = r"F:\projects\ai\pointnet\dataset\DMUNet_OBJ_format\unseen_data\Gear_1.pcd"
pred = infer_single_model(model, model_to_infer)
classID = torch.argmax(pred, dim=1)
className = classLabels[classID]
print(f"Predictions: {classID} : {className}")
