import numpy as np
import os

INPUT_PATH = "../../data/2"
first_file = True
for root, subdir, files in os.walk(INPUT_PATH):
    file_count = 0
    for file_name in files:
        file_count += 1
        if first_file is True:
            raw_data = np.loadtxt(root + "/" + file_name, delimiter=",")
            first_file = False
        else:
            raw_data = np.append(raw_data, np.loadtxt(root + "/" + file_name, delimiter=","), axis=0)
        print("finish sub_file:%d" % file_count)
    print ("finish folder" + root)
np.savez_compressed("zNup", data=raw_data)
