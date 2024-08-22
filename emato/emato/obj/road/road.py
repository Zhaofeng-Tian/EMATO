from emato.obj.poly.frenet_old import FrenetPath, generate_target_course
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import math

class Road:
    def __init__(self, wx, wy, num_lanes, lane_width=4):
        self.num_lanes = num_lanes
        self.lane_width = lane_width
        self.road_width = self.num_lanes * self.lane_width
        self.wx = wx
        self.wy = wy
        self.ts, self.tx, self.ty, self.tyaw, self.tk, self.csp = generate_target_course(wx, wy)

    def frenet_to_global(self, s, d):
        # index = min(range(len(self.ts)), key=lambda i: abs(self.ts[i] - s))
        # ts = 0,0.1,0.2, discretized by 0.1, so index is
        s = np.asarray(s)
        d = np.asarray(d)
        index = (s/0.1).astype(int)
        x_base = self.tx[index]
        y_base = self.ty[index]
        yaw_base = self.tyaw[index]
        x = x_base + d * np.cos(yaw_base + np.pi / 2)
        y = y_base + d * np.sin(yaw_base + np.pi / 2)
        yaw = yaw_base
        return x, y, yaw

    def plot_road(self, ax):
        lane_lines = self.create_lane_lines()
        for line in lane_lines:
            ax.plot(line['x'], line['y'], color='gray', linestyle=line['style'])

    def create_lane_lines(self):
        lane_lines = []
        half_road_width = (self.num_lanes * self.lane_width) / 2.0
        offsets = np.arange(-half_road_width, half_road_width + self.lane_width, self.lane_width)

        for i, offset in enumerate(offsets):
            lane_x = np.array(self.tx) + offset * np.cos(self.tyaw + np.pi / 2)
            lane_y = np.array(self.ty) + offset * np.sin(self.tyaw + np.pi / 2)
            lane_lines.append({
                "x": lane_x,
                "y": lane_y,
                "style": 'solid' if i == 0 or i == len(offsets) - 1 else 'dashed'
            })

        return lane_lines

if __name__ == "__main__":
    wx = [0.0,  350.0, 700, 1300, 1700]
    wy = [0.0, -100.0, 150.0, 65, 0.0]
    road = Road(wx, wy, num_lanes = 2, lane_width=4)
    x,y,yaw = road.frenet_to_global(np.array([100,200,300]), np.array([2,4,6]))
    print(x,y,yaw)