try:
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter

    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

def convolve_image(kernal_path,image,min_num_for_tfl_point):

    color_img = np.asarray(Image.open(kernal_path))/255
    array = np.mean(color_img, axis=2)
    array[array < 0.6] = array[array < 0.6] * -1
    kernel = np.array(array)
    image_after_convolve = sg.convolve(image, kernel, mode='same')
    max_light_points= maximum_filter(image_after_convolve, size=200) == image_after_convolve
    min_tfl_point=image_after_convolve>=min_num_for_tfl_point
    matrix_light_points=np.logical_and(min_tfl_point,max_light_points)
    return matrix_light_points,kernel

def n_max_values(mat, n):
    res_mat = np.zeros(mat.shape, dtype=bool)
    arr = mat.flatten()
    indices = np.argpartition(arr, -n)[-n:]
    n_max_indexes = np.vstack(np.unravel_index(indices, mat.shape)).T
    rows = n_max_indexes[:, 0]
    cols = n_max_indexes[:, 1]
    res_mat[rows, cols] = True
    return res_mat


# def create_kernal(size):
#
#     # create a circle kernel to detect circles
#     # the kernel
#
#     scharr = np.zeros(size ** 2).reshape(size, size)
#     # circle definitions
#     center = (int(size / 2), int(size / 2))
#     radius = int(size / 2)
#     count_pix_in_cir = 0
#     count_pix_out_cir = 0
#
#     for i in range(center[0] - radius, center[0] + radius + 1):
#         for j in range(center[1] - radius, center[1] + radius + 1):
#             if (center[0] - i) ** 2 + (center[1] - j) ** 2 <= radius ** 2:
#                 count_pix_in_cir += 1
#             else:
#                 count_pix_out_cir += 1
#     value_in = 2 / count_pix_in_cir
#     value_out = -1 / count_pix_out_cir
#
#     for i in range(center[0] - radius, center[0] + radius + 1):
#         for j in range(center[1] - radius, center[1] + radius + 1):
#             if (center[0] - i) ** 2 + (center[1] - j) ** 2 <= radius ** 2:
#                 scharr[i, j] = value_in
#             else:
#                 scharr[i, j] = value_out
#
#     b = np.array([[value_in * 0.7] for i in range(size)])
#     scharr = np.hstack((b, scharr, b))
#
#     return scharr

def find_tfl_lights(c_image: np.ndarray,pathImg,some_threshold=None):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    color_img = np.asarray(Image.open(pathImg)) / 255
    image=np.mean(color_img, axis=2)
    # light_points_1, kernel = convolve_image('./tryk.png', image, 20)
    light_points_1,kernel=convolve_image('./ker_1.png', image, 20)

    light_points_2,kernel=convolve_image('./ker_2.png',image,16)

    light_points=np.logical_or(light_points_2,light_points_1)

    light_points[1:10] = False
    light_points[:, -10:] = False
    light_points[:, 1:10] = False
    light_points[-50:] = False

    red = c_image[:, :, 0].astype(float)
    green = c_image[:, :, 1].astype(float)
    blue = c_image[:, :, 2].astype(float)

    red_light_points=np.logical_and(light_points, red >= green,red >= blue)
    green_light_points = np.logical_and(light_points, green > red , green > blue)

    red_lights_y,red_lights_x = np.where(red_light_points)
    green_lights_y, green_lights_x = np.where(green_light_points)

    return list(red_lights_x), list(red_lights_y), list(green_lights_x), list(green_lights_y)


### CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, red_x, red_y, green_x, green_y):
    plt.imshow(image)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]
    red_x, red_y, green_x, green_y = find_tfl_lights(image, image_path,some_threshold=42)
    show_image_and_gt(image, objects,red_x, red_y, green_x, green_y, fig_num)



def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = './data'
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)

def main_part_1(image_path):
    image = np.array(Image.open(image_path))
    return find_tfl_lights(image,image_path,some_threshold=42)