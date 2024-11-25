import time
import cv2
from pynput.mouse import Button, Controller
from grid import Grid
from screen import Screen
from util import find_items

mouse = Controller()


class Game():

    def __init__(self):
        self.screen = Screen()
        img = self.screen.get_image()
        self.mygrid = Grid(img)

    def run(self):
        """
        Runs the game for 160 seconds
        """

        start_time = time.time()
        step_number = 0
        while True:
            self.switch_gates(step_number)
            step_number+=1
            if time.time()-start_time > 160:
                break
        print('end')

    def find_path(self, start, goal, way):
        """Append a cell to the path

        Args:
            start (int): grid number of last cell
            goal (int): grid number of target cell
            way (list): list of grid number already part of the path

        Returns:
            list: list of grid numbers
        """
        if len(way) > 15:
            return False
        if start == goal:
            return way
        if self.mygrid.points[start]['type'] == 'station':
            return False
        way.append(start)
        for direction in self.mygrid.points[start]['directions']:
            res = self.find_path(direction, goal, way)
            if res is not False:
                return res
        del way[-1]
        return False

    def find_path_init(self, train):
        """Initializes path finding for a train

        Args:
            train

        Returns:
            array: list of cells to pass
            goal: target cell
        """
        start = train[0]
        goal = -1
        for idx, point in enumerate(self.mygrid.points):
            if point['type'] == 'station' and train[2] == point['color']:
                goal = idx
        if goal < 0:
            return False, False
        way = self.find_path(start, goal, [])
        if way is not False and train[1]-int(train[1]) > 0.4:
            way = way[1:]
        return way, goal


    def switch_gates(self, step):
        """Performs one step

        Args:
            step (int): step number for debugging
        """
        f2 = open('log.dat', 'a')
        f2.write('--'+str(step)+'--\n')
        start_time = time.time()

        img = self.screen.get_image()

        sec1_time = time.time()
        f2.write(str(sec1_time-start_time)+'\n')

        trains, n_stations = find_items(img, stations=False)
        if n_stations != self.mygrid.n_stations:
            f2.write('wrong number of stations detected\n')
            f2.close()
            return
        #cv2.imwrite('pic/'+str(step)+'.png', img)
        trainlist = []

        for train in trains:
            grid_nr = self.mygrid.get_grid_nr(train)
            distance = self.mygrid.get_distance(train)
            if grid_nr == 'error':
                continue
            if train['color'] == 'unknown':
                continue
            if self.mygrid.points[grid_nr]['type'] != 'station' and grid_nr != 1:
                trainlist.append([grid_nr, distance, train['color']])

        trainlist.sort(key=lambda x: x[1])

        for train in trainlist:
            path, goal = self.find_path_init(train)
            if path is not False:
                gates_pass = path
                goals_pass = path[1:]+[goal]
                f2.write(str(train)+'\n')
                f2.write(str(gates_pass)+'\n')
                for idx in range(len(gates_pass)):
                    self.mygrid.points[gates_pass[idx]]['state'] = goals_pass[idx]

        for point in self.mygrid.points:
            if point['type'] == 'gate':
                if point['state'] != point['ostate']:
                    f2.write('switch gate '+str(point['grid_nr'])+' from '+str(point['ostate'])+' to '+str(point['state'])+'\n')
                    mouse.position = (point['x']*self.screen.scale+self.screen.x, point['y']*self.screen.scale+self.screen.y)
                    mouse.click(Button.left, 1)
                    point['ostate'] = point['state']

        end_time = time.time()
        f2.write(str(end_time-sec1_time)+'\n')

        f2.close()


game = Game()

f1 = open('log.dat', 'w')
f1.write(str(game.mygrid.grid))
f1.write(str(game.mygrid.points))
f1.close()

game.run()
