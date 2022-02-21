import matplotlib.pyplot as plt
from part_1 import show_image_and_gt
from SFM_standAlone import visualize

class Veiw:
    def __init__(self):
        pass

    def veiw_frame(self, part_1,part_2, part_3):
        fig = plt.figure(figsize=(8, 12))
        fig.add_subplot(3, 1, 1)

        show_image_and_gt(part_3["curr_frame"].img, None, part_1["red_x"], part_1["red_y"], part_1["green_x"],
                          part_1["green_y"])
        plt.axis('off')
        plt.title("part1")
        fig.add_subplot(3, 1, 2)
        red_part_2 = [[],[]]
        green_part_2 = [[],[]]
        if len(part_2["red"]) > 0:
            red_part_2=[part_2["red"][:,0],part_2["red"][:, 1]]
        if len(part_2["green"]) > 0:
            green_part_2=[part_2["green"][:,0],part_2["green"][:, 1]]
        show_image_and_gt(part_3["curr_frame"].img, None, red_part_2[0],
                          red_part_2[1], green_part_2[0],
                          green_part_2[1])
        plt.axis('off')
        plt.title("part2")
        fig.add_subplot(3, 1, 3)
        visualize(part_3["prev_frame"],  part_3["curr_frame"],part_3["focal"], part_3["pp"],
                  part_3["prev_index"], part_3["curr_index"])
        plt.axis('off')
        plt.title("part3")
        plt.show()