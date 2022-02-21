import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from SFM_standAlone import FrameContainer, visualize
from part_1 import main_part_1
from part_2 import main_part_2
from SFM import calc_TFL_dist

class TFL_manager:
    def __init__(self,pkl_path):
        self.pkl_path = pkl_path
        self.curr_frame = None
        self.prev_frame = None
        with open(pkl_path, 'rb') as pklfile:
            self.data = pickle.load(pklfile, encoding='latin1')
        self.focal, self.pp = self.data['flx'], self.data['principle_point']
        self.prev_frame_id=None
        self.curr_frame_id=None
    def run_frame(self,curr_frame_id,frame_path):
        self.curr_frame_id = curr_frame_id
        self.curr_frame=FrameContainer(frame_path)
        """part1"""
        red_x, red_y, green_x, green_y = main_part_1(frame_path)
        part_1={"red_x":red_x, "red_y":red_y, "green_x":green_x, "green_y":green_y}
        """part2"""
        red_tfl_points, green_tfl_points = main_part_2(frame_path,red_x, red_y, green_x, green_y)
        self.curr_frame.traffic_light=red_tfl_points + green_tfl_points

        """part3"""
        if len(self.curr_frame.traffic_light)!=0 and self.prev_frame:
            EM = np.eye(4)
            for i in range(self.prev_frame_id, self.curr_frame_id):
                EM = np.dot(self.data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
            self.curr_frame.EM = EM
            self.curr_frame = calc_TFL_dist(self.prev_frame, self.curr_frame, self.focal, self.pp)
            # visualize(self.prev_frame, self.curr_frame, self.focal, self.pp,self.prev_frame_id,self.curr_frame_id)

        self.prev_frame = self.curr_frame
        self.prev_frame_id = curr_frame_id
        part_3={"curr_frame":self.curr_frame,
        "prev_frame":self.prev_frame,"prev_index":self.prev_frame_id,
                "curr_index":curr_frame_id,"focal":self.focal, "pp":self.pp}

        return part_1,{"red":np.array(red_tfl_points) , "green":np.array(green_tfl_points)},part_3






# import scipy
# from scipy import ndimage
# from scipy import signal
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(None).clf()
# imb = np.array(Image.open("img.png")).astype(float)
# ker = np.array(Image.open("ker.png")).astype(float)
# ker=[[1,1,1],[1,1,1],[1,1,1]]
# ker=[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]
# plt.imshow(imb)
# plt.imshow(ker)
# print(ker,imb)
# im=scipy.signal.convolve(imb,ker[0],mode="same")
# plt.imshow(im)
# im.show()
# image = np.array(im)
# im.show()
# print(image)
# iu=scipy.ndimage.maximum_filter(image,3)
# print(iu)
# plt.show(block=True)

# try:
#     import os
#     import json
#     import glob
#     import argparse
#     import numpy as np
#     from scipy import signal as sg
#     from scipy.ndimage.filters import maximum_filter
#
#     from PIL import Image
#
#     import matplotlib.pyplot as plt
# except ImportError:
#     print("Need to fix the installation")
#     raise


# def find_tfl_lights(c_image: np.ndarray,**kwargs):
#     """
#     Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
#     :param c_image: The image itself as np.uint8, shape of (H, W, 3)
#     :param kwargs: Whatever config you want to pass in here
#     :return: 4-tuple of x_red, y_red, x_green, y_green
#     """
#     ### WRITE YOUR CODE HERE ###
#     ### USE HELPER FUNCTIONS ###
#     # gray_image =
#     # print(gray_image)
#     print(c_image)
#     # np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
#
#
#     # kernel=np.array([0,0,0,0,0,0,0,0,0,0,0,
#     #   0,0,0,0,0,0.40,0,0,0,0,0,
#     #   0,0,0,0,0.40,0.55,0.40,0,0,0,0,
#     #   0,0,0,0.40,0.55,0.7,0.55,0.40,0,0,0,
#     #   0,0,0.40,0.55,0.7,0.85,0.7,0.55,0.40,0,0,
#     #   0,0.40,0.55,0.7,0.85,1,0.85,0.7,0.55,0.40,0,
#     #   0,0,0.40,0.55,0.7,0.85,0.7,0.55,0.40,0,0,
#     #   0,0,0,0.40,0.55,0.7,0.55,0.40,0,0,0,
#     #   0,0,0,0,0.40,0.55,0.40,0,0,0,0,
#     #   0,0,0,0,0,0.40,0,0,0,0,0,
#     #   0,0,0,0,0,0,0,0,0,0,0])
#     kernel=np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0]])
#     sg.convolve(c_image, kernel, mode='same')
#     print("llllllllllllllllllllll")
#     print(c_image)
#
#     return 100,150,102,180


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
# def show_image_and_gt(image, objs, fig_num=None):
#     plt.figure(fig_num).clf()
#     plt.imshow(image,cmap=plt.get_cmap(name='gray'))
#
#     labels = set()
#     if objs is not None:
#         for o in objs:
#             poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
#             plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
#             labels.add(o['label'])
#         if len(labels) > 1:
#             plt.legend()


# def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
#     """
#     Run the attention code
#     """
#     # image = np.array(Image.open(image_path).convert('L'))
#     color_img = np.asarray(Image.open(image_path)) / 255
#     image = np.mean(color_img, axis=2)
#     print(image)
#     if json_path is None:
#         objects = None
#     else:
#         gt_data = json.load(open(json_path))
#         what = ['traffic light']
#         objects = [o for o in gt_data['objects'] if o['label'] in what]
#     show_image_and_gt(image, objects, fig_num)
#
#
#     red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
#     plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
#     plt.plot(green_x, green_y, 'ro', color='g', markersize=4)



# def main(argv=None):
#     """It's nice to have a standalone tester for the algorithm.
#     Consider looping over some images from here, so you can manually exmine the results
#     Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
#     :param argv: In case you want to programmatically run this"""
#
#     parser = argparse.ArgumentParser("Test TFL attention mechanism")
#     parser.add_argument('-i', '--image', type=str, help='Path to an image')
#     parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
#     parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
#     args = parser.parse_args(argv)
#     default_base = './leftImg8bit_trainvaltest/leftImg8bit/train/jena/'
#
#     if args.dir is None:
#         args.dir = default_base
#     flist = glob.glob(os.path.join(args.dir, 'jena_000029_000019_leftImg8bit.png'))
#
#     for image in flist:
#         json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
#
#         if not os.path.exists(json_fn):
#             json_fn = None
#         test_find_tfl_lights(image, json_fn)
#
#     if len(flist):
#         print("You should now see some images, with the ground truth marked on them. Close all to quit.")
#     else:
#         print("Bad configuration?? Didn't find any picture to show")
#     plt.show(block=True)


# if __name__ == '__main__':
#     main()