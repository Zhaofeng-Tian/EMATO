import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from emato.obj.road.road import Road


class TrafficCar:
    def __init__(self, lane, sd, speed, road,length=4.5, width=2.0):
        self.lane = lane
        self.speed = speed
        self.length = length
        self.width = width
        self.history_length = 100
        self.position_history = []
        self.yaw_history = []
        s,d = sd
        # d= self.lane*road.lane_width - road.lane_width
        # d= self.lane*road.lane_width
        d = -road.road_width/2 + road.lane_width/2 + self.lane * road.lane_width
        print(" d: {}, road width: {}, lane width: {}, my lane: {}")
        self.s = s  # Frenet longitudinal coordinate
        self.d = d  # Frenet lateral offset
        self.road = road
        x, y, yaw = self.road.frenet_to_global(s, d)
        self.x =x; self.y = y; self.yaw = yaw
        self.pose_history = [(x, y,yaw)]
        self.furture_sd_points = []
        self.future_traj = []


    def traffic_update_position(self, dt):
        """
        Update the simulated traffic car with constant speed
        """
        self.s += self.speed * dt
        x, y, yaw = self.road.frenet_to_global(self.s, self.d)
        self.pose_history.append((x, y, yaw))
        self.x =x; self.y = y; self.yaw = yaw
        if len(self.pose_history) > self.history_length:
            self.pose_history.pop(0)

    def calc_future_sd_points(self, ptime, dt):
        """
        Only for traffic cars
        ptime: prediction time
        """

        self.furture_sd_points = []
        self.furture_sd_points = []
        for t in range(0, int(ptime / dt)):
            future_s = self.s + self.speed * t * dt
            self.furture_sd_points.append((future_s, self.d))
    
    def set_traj(self, future_sd_points):
        self.furture_sd_points = future_sd_points
        self.future_traj = [self.road.frenet_to_global(s, d) for s, d in future_sd_points]
        self.x, self.y, self.yaw = self.future_traj[0]
        self.pose_history.append(self.future_traj[0])
    
    def set_1d_traj(self, future_s_points):
        self.future_traj = [(s,0,0) for s in future_s_points]
        self.x, self.y, self.yaw = self.future_traj[0]
        self.pose_history.append(self.future_traj[0])
        
    def get_rotated_corners(self):
        # Get the most recent position and yaw
        x, y, yaw = self.pose_history[-1]


        # Define the unrotated rectangle corners relative to the car's center
        half_length = self.length / 2.0
        half_width = self.width / 2.0
        corners = np.array([
            [half_length, half_width],  # front-left
            [half_length, -half_width],   # front-right
            [-half_length, -half_width],    # rear-right
            [-half_length, half_width]    # rear-left
        ])

        # Rotation matrix based on yaw angle
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])

        # Rotate and translate corners to global position
        # rotated_corners = np.dot(corners, rotation_matrix) + [x, y]
        rotated_corners = corners@rotation_matrix.T + [x,y]
        # print(" corners: ", rotated_corners)
        return rotated_corners
    
    def get_future_sd_points(self):
        return np.array(self.furture_sd_points)



def plot_cars(ax, cars):
    for car in cars:
        rotated_corners = car.get_rotated_corners()
        rect = Polygon(rotated_corners, edgecolor='black', facecolor='lightblue')
        ax.add_patch(rect)
        x, y = car.position_history[-1]
        yaw = car.yaw_history[-1]
        ax.arrow(
            x, y,
            car.speed * np.cos(yaw),
            car.speed * np.sin(yaw),
            head_width=0.5, head_length=1.0, fc='red', ec='red', alpha=0.7
        )

def simulate(road, cars, total_time, time_step):
    fig, ax = plt.subplots(figsize=(10, 5))
    window_width = 30
    
    for t in np.arange(0, total_time, time_step):
        ax.clear()
        road.plot_road(ax)
        for car in cars:
            car.update_position(time_step, road)
        print("car1 s,d: ", cars[0].s, cars[0].d)
        plot_cars(ax, cars)
        ax.set_xlim([cars[0].position_history[-1][0]-window_width, cars[0].position_history[-1][0]+window_width])
        ax.set_ylim([cars[0].position_history[-1][1]-window_width, cars[0].position_history[-1][1]+window_width])
        ax.set_aspect('equal')
        ax.set_title(f"Highway Simulation at t={t:.2f} s")
        plt.pause(0.01)

    plt.show()


def main():
    wx = [0.0, 350.0, 700.0, 1300.0, 1700.0]
    wy = [0.0, 500.0, 150.0, 65.0, 0.0]

    road = Road(wx, wy, num_lanes=1)
    bs = 300 # base s
    # car1 = Car(lane=1, sd= [bs+30,0], speed=13.0, road = road)
    # car2 = Car(lane=0, sd = [bs+0,0], speed=12.0 ,road = road)
    # car3 = Car(lane=0, sd = [bs+20,0], speed=12.0 ,road = road)
    # car4 = Car(lane=0, sd = [bs+45,0], speed=12.0 ,road = road)
    # car5 = Car(lane=2, sd = [bs+0,0], speed=10.0 ,road = road)
    # car6 = Car(lane=2, sd = [bs+20,0],speed=11.0 ,road = road)
    # car7 = Car(lane=2, sd = [bs+50,0],speed=12.0 ,road = road)

    

    # cars = [car1, car2, car3, car4, car5, car6, car7]
    car1 = TrafficCar(lane=0, sd= [bs+30,0], speed=13.0, road = road)
    cars = [car1]
    total_time = 50.0
    time_step = 0.1

    simulate(road, cars, total_time, time_step)

if __name__ == "__main__":
    main()
 
