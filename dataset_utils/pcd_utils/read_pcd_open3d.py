# SandboxScript: Script to Read and Write PCD file using Open-3D Library

import open3d as o3d
import numpy as np
import webcolors as wc

# Read a PCD file 
#filePath = r"F:\projects\ai\pointnet\dataset\unseen_models_segmentation\chair_1_ascii.pcd"
#filePath = r"F:\projects\ai\pointnet\dataset\seg_models\screw_head_ascii.pcd"
filePath = r"F:\projects\ai\pointnet\dataset\seg_models\screw_merged_scalar.pcd"

cloud = o3d.io.read_point_cloud(filePath)

#o3d.visualization.draw_geometries([cloud])

xyz = np.asarray(cloud.points)
rgb = np.asarray(cloud.colors)

print(f"Shape Points: {xyz.shape}")
print(f"{xyz}")
print(f"Shape Points: {rgb.shape}")
print(f"{rgb}")

print("done")

#######################################################################
# #Creating colors matrix
# def create_row():
#     red_color = wc.name_to_rgb('green')
#     #print(red_color)

#     red_array = np.asarray(red_color) / 255.0
#     #print(red_array )
#     return red_array
    
# color_array = np.empty((10, 3))
#print(wc.name_to_rgb('cyan'))
# for i in range(10):
#     color_array[i] = create_row()

# print(color_array.shape)
# print(color_array)

#######################################################################
# # Writing a PCD File

# # generate some neat n times 3 matrix using a variant of sync function
# x = np.linspace(-3, 3, 401)
# mesh_x, mesh_y = np.meshgrid(x, x)
# z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
# z_norm = (z - z.min()) / (z.max() - z.min())
# xyz = np.zeros((np.size(mesh_x), 3))
# xyz[:, 0] = np.reshape(mesh_x, -1)
# xyz[:, 1] = np.reshape(mesh_y, -1)
# xyz[:, 2] = np.reshape(z_norm, -1)
# print('xyz')
# print(xyz.shape)

# #create color matrix
# #the below code is time consuming, need to find a way to make this faster
# color_matrix = np.empty((xyz.shape[0], 3))
# for i in range(xyz.shape[0]):
#     color_matrix[i] = create_row()

# # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz)
# pcd.colors = o3d.utility.Vector3dVector(color_matrix)
# o3d.io.write_point_cloud(r"F:\projects\ai\pointnet\dataset\seg_models\open3d.pcd", pcd, write_ascii=True)
# print('done')

#######################################################################
## Genereate Image

img = o3d.geometry.Image(   )

