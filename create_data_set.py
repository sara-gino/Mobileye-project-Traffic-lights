import math
import statistics

import matplotlib.pyplot as plt
from PIL import Image
from numpy import uint8
import numpy as np
import cv2
import glob
import random

def cut_img(pat_image,point):
    im = Image.open(pat_image)
    left = max(0, point[1] - 40)
    top = max(0, point[0] - 40)
    bottom = min(1024, top + 81)
    right = min(2047, left + 81)
    return uint8(np.asarray(im.crop((left, top, right, bottom))))


def write_bin(img: np.ndarray, is_traffic,output):
    with open(f"./{output}/data.bin", "ba+") as data_file:
        with open(f"./{output}/labels.bin", "ba+") as labels_file:
            img.tofile(data_file)
            labels_file.write(uint8(is_traffic))


def create_set(dir_path, output):
    files = []
    files.extend(glob.glob(dir_path + '*labelIds.png'))
    for file in files:
        image = uint8(cv2.imread(file))
        result =np.where(image==19)
        listOfCoordinates = list(zip(result[0], result[1]))
        source_file = file.replace('gtFine/', 'leftImg8bit/')
        source_file = source_file.replace('gtFine_labelIds', 'leftImg8bit')
        random_cord=0
        while len(listOfCoordinates)>0:
            tuple_min = 0
            start_point = (listOfCoordinates[0][0], listOfCoordinates[0][1])
            tfl_points=[]
            for coordinate in listOfCoordinates:
                if math.dist(coordinate, start_point) <= 50:# coordinate points in a tfl 
                    tfl_points.append(coordinate)
                    ind=listOfCoordinates.index(coordinate)
                    listOfCoordinates=listOfCoordinates[:ind]+listOfCoordinates[ind+1:]
            X = [tuple[0] for tuple in tfl_points]
            Y = [tuple[1] for tuple in tfl_points]
            tuple_min = (statistics.mean(X),statistics.mean(Y))
            img=cut_img(source_file, tuple_min)
            if tuple_min!=0 and img.shape==(81,81,3):
                write_bin(img, 1, output)
                not_traffices = np.where(image != 19)
                cordinates = list(zip(not_traffices[0], not_traffices[1]))
                not_found = True
                while not_found:
                       random_cord = random.choice(cordinates)
                       not_traffic = np.asarray(cut_img(file, random_cord))
                       img = cut_img(source_file, random_cord)
                       if 19 not in not_traffic and img.shape == (81, 81, 3):
                           not_found = False
                if random_cord!=0:
                     write_bin(img, 0, output)


"""create a data set-bin file to train our machine"""
# def create_data_set():
#     create_set('gtFine/train/*/', 'train')
#     create_set('gtFine/val/*/', 'val')
# create_data_set()

"""in a Team worer each poeple create bin file on some data and after that we union them"""
# def Union_2_bin_file(file1,file2):
#     with open(file1, "ba+") as myfile, open(file2, "rb") as file2:
#         myfile.write(file2.read())
# my_file_data="union_train/data.bin"
# my_file_load="union_train/labels.bin"
# sari_data="sari_train/data.bin"
# sari_load="sari_train/labels.bin"
# Union_2_bin_file(my_file_data,sari_data)
# Union_2_bin_file(my_file_load,sari_load)
# rachel_data="racely_train/data.bin"
# rachel_load="racely_train/labels.bin"
# Union_2_bin_file(my_file_data,rachel_data)
# Union_2_bin_file(my_file_load,rachel_load)


