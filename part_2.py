from create_data_set import cut_img
from tensorflow.keras.models import load_model
import numpy as np

loaded_model = load_model("./model.h5")

def verify_Tfls(image_path,array_x, array_y):
    array_images = []
    Tfls_points = []
    for i in range(len(array_x)):
        img_val = cut_img(image_path, (array_y[i], array_x[i]))
        array_images.append(img_val)

    array_images = np.asarray(array_images)
    l_predictions = []
    if len(array_images)>0:
        l_predictions = loaded_model.predict(array_images)
    for i in range(len(l_predictions)):
        if l_predictions[i][1] > 0.85:
            Tfls_points.append([array_x[i], array_y[i]])
    return Tfls_points

def main_part_2(image_path,red_x, red_y,green_x, green_y):

    green_tfl_points = verify_Tfls(image_path,green_x, green_y)

    red_tfl_points = verify_Tfls(image_path,red_x, red_y)

    return red_tfl_points , green_tfl_points