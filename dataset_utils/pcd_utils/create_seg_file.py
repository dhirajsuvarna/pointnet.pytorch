# Script to generate segementation file containing lables "fileName.seg" from the "pcd" file
# This script expects the PCD file
#           - to be in ASCII format 
#           - to have FIELDS format as 
#               FIELDS x y z label object 
#            
# todo: need to handle rgb values if present in the file

import os
import sys
import argparse

def validate_fields(line):
    fields = line.split()
    if fields[1] == 'x' and fields[2] == 'y' and fields[3] == 'z' and fields[4] == 'label':
        return True
    else:
        return False

def validate_data(line):
    dataValues = line.split()
    if dataValues[1] == 'ascii':
        return True
    else:
        return False

def read_header(pcdFile):
    for line in pcdFile:
        if "FIELDS" in line:
            isFieldValid = validate_fields(line)
        elif "DATA" in line: #validate data 
            isAscii = validate_data(line)
            break

    if isFieldValid and isAscii:
        return True
    else:
        return False

def read_label_data(pcdFile):
    lableData = []
    for lineNo, line in enumerate(pcdFile):
        data = line.split()
        lableData.append(data[3]) #it is expected that label is in the 4th poisition in the line
        
    return lableData

def write_seg_file(labelData, segFilePath):
    print(f"Writing file: {segFilePath}")
    with open(segFilePath, 'w') as segFile:
        for label in labelData:
            segFile.write("%s\n" %label)



def generate_seg_file(pcdFilePath):
    with open(pcdFilePath) as pcdFile:
        isValidHeader = read_header(pcdFile)
        if not isValidHeader:
            print("Unable to read header")
            sys.exit()

        labelData = read_label_data(pcdFile)

        segFilePath = pcdFilePath.replace('.pcd', '.seg')
        write_seg_file(labelData, segFilePath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Input Folder containing PCD files or Single PCD File")
    #parser.add_argument("--output", required=False, help="Output Folder to generate .seg files, if not provided Default will be input folder")

    args = parser.parse_args()
    print(args)

    input = args.input
    #output = args.output

    if os.path.isdir(input):
        for root, subdirs, files in os.walk(input):
            for fileName in files:
                if fileName.endswith('.pcd'):
                    filePath = os.path.join(root, fileName)
                    generate_seg_file(filePath)

    elif os.path.isfile(input):
        if not input.endswith('.pcd'):
            print(f"{input}: Invalid input format, .pcd file is expected")
            sys.exit()
        
        generate_seg_file(input)