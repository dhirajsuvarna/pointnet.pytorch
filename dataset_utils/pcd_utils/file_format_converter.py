# Script to convert various file format to Point Cloud Data (PCD)

import os
import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument("--input_folder", required=True, help="Input folder containing files")
parser.add_argument("--output_folder", required=True, help="Output folder where the files are generated")
parser.add_argument("--input_format", required=True, choices=["stl", "obj"], help="File format of input files")

args = parser.parse_args()

inputFolder = args.input_folder
outputFolder = args.output_folder
inputFormat = args.input_format

pcl_converter = r"C:\Program Files\PCL 1.10.0\bin\pcl_converter.exe"

for root, subdirs, files in os.walk(inputFolder):
    for fileName in files:
        filePath = ""
        pcdFilePath = ""
        if inputFormat == "stl" and fileName.lower().endswith(".stl") :
            filePath = os.path.join(root, fileName)
            pcdFilePath = filePath.replace(inputFolder, outputFolder).replace(".stl", ".pcd")
        elif inputFormat == "obj" and fileName.lower().endswith(".obj"):
            filePath = os.path.join(root, fileName)
            pcdFilePath = filePath.replace(inputFolder, outputFolder).replace(".obj", ".pcd")
        else:
            continue
        
        os.makedirs(os.path.dirname(pcdFilePath), exist_ok=True)

        commandList = [pcl_converter, "-f", "ascii", filePath, pcdFilePath]
        print(f"Processing File: {filePath}")
        subprocess.run(commandList)