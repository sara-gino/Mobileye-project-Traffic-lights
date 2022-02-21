from tfl_manager import TFL_manager
from View import Veiw
class Controller:
    def __init__(self, playlist):
        with open(playlist, "r") as playlist:
            self.playlist = playlist.readlines()
        self.tfl_manager = TFL_manager(self.playlist[0][:-1])  # self.playlist[0] == pkl file
        self.view=Veiw()

    def run(self):
        for i in range(len(self.playlist)-2):
            tfl_part1,tfl_part2, tfl_part3 = self.tfl_manager.run_frame(i + int(self.playlist[1][:-1]), self.playlist[i+2][:-1])
            if i>0:
                self.view.veiw_frame(tfl_part1, tfl_part2,tfl_part3)

controller = Controller("./play_list.pls")
controller.run()