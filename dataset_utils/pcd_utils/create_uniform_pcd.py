import os
import subprocess
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--mesh_sampling_path", required=True, help="Path to the pcl_mesh_sampling.exe")
parser.add_argument("--n_samples", default=5000, type=int,  help="Number of points required in the Point Cloud Data")
parser.add_argument("--input_dir", required=True, help="Path to input directory")
parser.add_argument("--output_dir", required=True, help="Path to output directory")

args = parser.parse_args()

numOutputPoints = args.n_samples
exePath = args.mesh_sampling_path
inputDir = args.input_dir
outputDir = args.output_dir

options = ["-n_samples", str(numOutputPoints), "-leaf_size", "0.05", "-no_vis_result"]

for rootDir, subDirs, files in os.walk(inputDir):
    # for each obj file generate a pcd file with the given number of points 
    # number of point is from input
    for fileName in files:
        if(fileName.lower().endswith('obj')):
            filePath = os.path.join(rootDir, fileName)
            subDir = rootDir.split(inputDir, 1)[1].split("\\")[1]
            outputPath = os.path.join(outputDir, subDir)
            os.makedirs(outputPath, exist_ok=True)
            pcdPath = os.path.join(outputPath, fileName.replace('.obj', '.pcd'))

            processArgs = []
            processArgs.append(exePath)
            processArgs.append(filePath)
            processArgs.append(pcdPath)
            processArgs = processArgs + options
            subprocess.run(processArgs)



