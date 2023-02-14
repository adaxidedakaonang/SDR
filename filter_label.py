import os
from PIL import Image
import numpy as np
import glob
import shutil
from tqdm import tqdm

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

class FilterManager():
    def __init__(self, file, aim_class, size):
        self.file = self._toNumpy(file)
        self.aim_class = aim_class
        self.aim_size = size
        self.actual_size = None
        self.max_size = None
    
    def classExists(self):
        classes, counts = np.unique(self.file, return_counts=True)
        if len(counts)>1:
            self.max_size = np.max(counts[1:])
        for i in range(len(classes)):
            if self.aim_class==classes[i]:
                self.actual_size = counts[i]
                return True
        return False

    def sizeReach(self):
        r,c = self.file.shape
        total_size = r*c
        if self.actual_size / 1.0 / total_size >=self.aim_size:
            return True
        else:
            return False

    def sizeMax(self):
        if self.max_size is not None:
            if not self.actual_size == self.max_size:
                return False
            else:
                return True

    def _toNumpy(self, file):
        if not file is None:
            return np.array(file)
        else:
            return None


root_path = r"C:\ADAXI\Replay_Data"
tag = r"\label_sdr_10-10_non_reach_0.2"
# tag = r"\RECALL\labels"
classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
range_low = 0
range_high = 10
size = 0.1
class_path = os.listdir(root_path)[range_low:range_high]
classes = classes[range_low:range_high]
for idx in range(len(class_path)):
    print(class_path[idx])
# for c_path in class_path:
    src_path = os.path.join(root_path,class_path[idx] + tag)
    bad_path = src_path + "_non_exists"
    small_path = src_path + "_non_reach"
    max_path = src_path + "_non_max"
    aim_class = classes[idx]
    print(f"target class index: {aim_class}")
    
    small_path = small_path + "_" + str(size)
    create_folder(bad_path)
    create_folder(small_path)
    create_folder(max_path)
    total_rate = 0
    img_list = glob.glob( (os.path.join(src_path, "*.png")) )
    for _ in tqdm(img_list):
        filterManager = FilterManager(file=Image.open(_), aim_class=aim_class, size=size)
        img_name = os.path.split(_)[1]
        if not filterManager.classExists():
            shutil.move(_, os.path.join(bad_path, img_name) )
        elif not filterManager.sizeMax():
            shutil.move(_, os.path.join(max_path, img_name) )
        elif not filterManager.sizeReach():
            shutil.move(_, os.path.join(small_path, img_name) )

    print()
