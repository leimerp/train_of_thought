import numpy as np
import cv2
from util import find_items

class Grid:
    def __init__(self, img):
        self.grid = np.ones((7,5), dtype=np.uint8)*-1
        self.img = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        stations, self.n_stations = find_items(img, stations=True)

        gates = self.find_gates(img)
        self.points = stations + gates

        (height, width) = self.gray.shape
        min_x = width
        max_x = 0
        min_y = height
        max_y = 0
        for point in self.points:
            if 'x' not in point:
                continue
            if point['x'] < min_x:
                min_x = point['x']
            if point['x'] > max_x:
                max_x = point['x']
            if point['y'] < min_y:
                min_y = point['y']
            if point['y'] > max_y:
                max_y = point['y']

        min_x -= 15
        max_x += 15
        min_y -= 15
        max_y += 15
        self.min_y = min_y
        self.min_x = min_x
        self.gridsize_x = (max_x-min_x)//5
        self.gridsize_y = (max_y-min_y)//7

        for idx, point in enumerate(self.points):
            if 'x' in point:
                x, y=self.get_grid_position(point)
                self.grid[y, x] = int(idx)
                point['grid_nr'] = idx

        start = self.find_start()
        self.align_grid_positions()
        self.find_directions(len(self.points)-1, start, 2)


    def get_grid_position(self, point):
        """Get the grid position of a point

        Args:
            point

        Returns:
            int: x-position
            int: y-position
        """
        x_grid = (point['x']-self.min_x)//self.gridsize_x
        y_grid = (point['y']-self.min_y)//self.gridsize_y
        return x_grid, y_grid

    def align_grid_positions(self):
        """Aligns the coordinates of the grids
        """
        for y in range(self.grid.shape[0]):
            mean_y = []
            for idx in self.grid[y, :]:
                if idx < 0:
                    continue
                if 'y' in self.points[idx]:
                    mean_y.append(self.points[idx]['y'])
            if len(mean_y) > 0:
                for idx in self.grid[y, :]:
                    if idx < 0:
                        continue
                    self.points[idx]['y'] = round(np.mean(mean_y))

        for x in range(self.grid.shape[1]):
            mean_x = []
            for idx in self.grid[:, x]:
                if idx < 0:
                    continue
                if 'x' in self.points[idx]:
                    mean_x.append(self.points[idx]['x'])
            if len(mean_x) > 0:
                for idx in self.grid[:, x]:
                    if idx < 0:
                        continue
                    self.points[idx]['x'] = round(np.mean(mean_x))


    def get_grid_nr(self, train):
        """Get the grid number of a train

        Args:
            train

        Returns:
            int: grid number
        """
        x, y = self.get_grid_position(train)
        if x >= self.grid.shape[1] or y >= self.grid.shape[0]:
            return 'error'
        return self.grid[y, x]

    def get_distance(self, train):
        """Caculates the distance of a train from start

        Args:
            train

        Returns:
            float: distance between train and start
        """
        grid_nr = self.get_grid_nr(train)
        if grid_nr == 'error' or 'distance' not in self.points[grid_nr]:
            return 0
        dist = self.points[grid_nr]['distance']
        if 'x' in self.points[grid_nr]:
            x_dist = train['x']-self.points[grid_nr]['x']
            y_dist = train['y']-self.points[grid_nr]['y']
            if self.points[grid_nr]['incoming'] == 0: # top
                y_dist += 30
            elif self.points[grid_nr]['incoming'] == 1: # bottom
                y_dist -= 30
            elif self.points[grid_nr]['incoming'] == 2: # left
                x_dist += 30
            elif self.points[grid_nr]['incoming'] == 3: # right
                x_dist -= 30
            dist += np.sqrt(np.power(x_dist, 2)+np.power(y_dist, 2))/100
        return dist

    def find_start(self):
        """Find the start point (point of first gate)

        Args:
            points

        Returns:
            int: grid number of start point
        """
        start = None
        for column in range(self.grid.shape[1]):
            if self.grid[0][column] == -1 and self.grid[1][column] == -1 and self.grid[2][column] == -1:
                incoming = 0 # top
                self.grid[0][column] = len(self.points)
                self.grid[1][column] = len(self.points)+1
                self.grid[2][column] = len(self.points)+2
                if self.grid[-4][column] > -1 and self.points[self.grid[3][column]]['type'] == 'gate':
                    start = self.grid[3][column]
                elif column > 0 and self.points[self.grid[2][column-1]]['type'] == 'gate':
                    start = self.grid[2][column-1]
                elif column < len(self.grid[0,:])-1 and self.points[self.grid[2][column+1]]['type'] == 'gate':
                    start = self.grid[2][column+1]
            elif self.grid[-1][column] == -1 and self.grid[-2][column] == -1 and self.grid[-3][column] == -1:
                incoming = 1 # bottom
                self.grid[-1][column] = len(self.points)
                self.grid[-2][column] = len(self.points)+1
                self.grid[-3][column] = len(self.points)+2
                if self.grid[-4][column] > -1 and self.points[self.grid[-4][column]]['type'] == 'gate':
                    start = self.grid[-4][column]
                elif column > 0 and self.points[self.grid[-3][column-1]]['type'] == 'gate':
                    start = self.grid[-3][column-1]
                elif column < len(self.grid[0,:])-1 and self.points[self.grid[-3][column+1]]['type'] == 'gate':
                    start = self.grid[-3][column+1]


        self.points.append({'type': 'track', 'distance': 0, 'incoming': incoming, 'directions': [len(self.points)+1], 'state': len(self.points)+1, 'ostate': len(self.points)+1})
        self.points.append({'type': 'track', 'distance': 1, 'incoming': incoming, 'directions': [len(self.points)+1], 'state': len(self.points)+1, 'ostate': len(self.points)+1})
        self.points.append({'type': 'track', 'distance': 2, 'incoming': incoming, 'directions': [start], 'state': start, 'ostate': start})

        return start


    def find_directions(self, prev, gate_nr, distance):
        """Finds iteratively for each gate the two possible directions to continue

        Args:
            prev (int): grid number of the previous cell
            gate_nr (int): index of the gate
            distance (int): number of cells passed since start
        """
        distance += 1
        self.points[gate_nr]['distance'] = distance
        gate = self.points[gate_nr]
        pos = np.where(self.grid == gate_nr)
        pos_x = pos[1][0]
        pos_y = pos[0][0]
        len_x = len(self.grid[0,:])
        len_y = len(self.grid[:,0])

        directions = np.array([self.grid[pos_y-1, pos_x], self.grid[(pos_y+1)%len_y, pos_x], self.grid[pos_y, pos_x-1], self.grid[pos_y, (pos_x+1)%len_x]]) # [up, down, left, right]
        incoming = np.where(directions==prev)[0][0]

        padding = 45
        left_limit = np.max([gate['y']-padding, 0])
        right_limit = np.min([gate['y']+padding, self.gray.shape[0]])
        top_limit = np.max([gate['x']-padding, 0])
        bottom_limit = np.min([gate['x']+padding, self.gray.shape[1]])
        cropped = self.gray[left_limit:right_limit, top_limit:bottom_limit]
        _, thresh = cv2.threshold(cropped, 50, 255, cv2.THRESH_BINARY)
        thresh = 255-thresh

        if pos_y == 0:
            deadend = directions[0]
        elif pos_y == len_y-1:
            deadend = directions[1]
        elif pos_x == 0:
            deadend = directions[2]
        elif pos_x == len_x-1:
            deadend = directions[3]
        else:
            ml = [np.sum(thresh[0:20,:]), np.sum(thresh[-21:-1,:]), np.sum(thresh[:,0:20]), np.sum(thresh[:,-21:-1])]
            deadend = directions[np.argmin(ml)]

        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 60, np.array([]), 50, 0)
        if lines is not None:
            state = directions[(np.where(directions==prev)[0][0]+1)%2+np.where(directions==prev)[0][0]//2*2]
        else:
            if np.where(directions==prev)[0][0] < 2:
                if directions[2] == deadend:
                    state = directions[3]
                else:
                    state = directions[2]
            else:
                if directions[0] == deadend:
                    state = directions[1]
                else:
                    state = directions[0]

        directions = list(directions)
        directions.remove(prev)
        directions.remove(deadend)

        for direction in directions:
            self.points[gate_nr]['directions'].append(direction)
            self.points[gate_nr]['state'] = state
            self.points[gate_nr]['ostate'] = state
            self.points[gate_nr]['incoming'] = incoming
            if self.points[direction]['type'] == 'gate' and len(self.points[direction]['directions']) == 0:
                self.find_directions(gate_nr, direction, distance)


    def find_gates(self, img):
        """Find all gates in an image

        Args:
            img (RGB-image): screenshot of the game window

        Returns:
            list: list of gates
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gates = []
        gray_blurred = cv2.blur(gray, (3, 3))
        detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 50, param1 = 50, param2 = 40, minRadius = 22, maxRadius = 35)
        if detected_circles is not None:
            detected_circles = np.around(detected_circles).astype(np.uint16)
            for point in detected_circles[0, :]:
                c_x, c_y, _ = point[0], point[1], point[2]
                gates.append({'x': c_x, 'y': c_y, 'type': 'gate', 'directions': []})
        return gates
