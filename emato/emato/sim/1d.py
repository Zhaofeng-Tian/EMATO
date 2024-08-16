from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from emato.param.param import TruckParam
from emato.obj.road.road import Road
from emato.obj.car.car import Car
from emato.obj.car.traffic_car import TrafficCar
from emato.util.util import get_leading_profile, local_approx, get_gradient_profile, get_traj_theta
from emato.alg.nlp import NLP


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
    window_width = 10
    
    for t in np.arange(0, total_time, time_step):
        ax.clear()
        road.plot_road(ax)
        for car in cars:
            car.update_position(time_step, road)
        print("car1 s,d: ", cars[0].s, cars[0].d)
        plot_cars(ax, cars)
        ax.set_xlim([cars[0].position_history[-1][0]-window_width, cars[0].position_history[-1][0]+window_width*10])
        ax.set_ylim([cars[0].position_history[-1][1]-window_width, cars[0].position_history[-1][1]+window_width])
        ax.set_aspect('equal')
        ax.set_title(f"Highway Simulation at t={t:.2f} s")
        plt.pause(0.01)

    plt.show()


def main():
    # wx = [0.0, 350.0, 700.0, 1300.0, 1700.0]
    # wy = [0.0, 500.0, 150.0, 65.0, 0.0]
    wx = [0.0, 10000]
    wy = [0.0, 0.]
    road = Road(wx, wy, num_lanes=1)

    # Setting global parameters
    param = TruckParam(cycle = 'HWFET')
    param.prediction_time = 5 # predict the leading car's trajectory over 5 seconds horizon
    param.gd_profile_type = "rolling"

    # Section. A: Initialization
    # 1. Get leading car and gradient profile
    profile_leading = get_leading_profile(param.cycle)
    profile_altitude, profile_gradient = get_gradient_profile(feature=param.gd_profile_type)
    # 2. Init cars
    car_l = Car(param)
    car_e = Car(param)
    # 3. Local prediction of leading car
    traj_sl, traj_vl, traj_al = local_approx(profile_leading,time_target=param.ts, delta_time=param.prediction_time, dt = param.dt)
    print("traj_sl: ", traj_sl)
    print("traj_vl: ", traj_vl)
    print("traj_al: ", traj_al)
    # 4. Set inital states for both cars
    car_l.set_state(traj_sl[0],traj_vl[0], traj_al[0], theta= profile_gradient[int(traj_sl[0]- param.dinit)], fc = 0.)
    car_e.set_state(traj_sl[0] - param.dinit,traj_vl[0], traj_al[0], theta= profile_gradient[int(traj_sl[0]- param.dinit)], fc = 0.)
    # 5. Set up traffic cars for visualization
    tcar_l = TrafficCar(lane = 0, sd = [car_l.s,0], speed = 0.0, road = road)
    tcar_e = TrafficCar(lane = 0, sd = [car_e.s,0], speed = 0.0, road = road)
    # 6. Solver initialization
    solver = NLP(param, (traj_sl, traj_vl, traj_al), xc = car_e.get_x())
    # Section. B: Simulation step
    sim_time = param.ts
    global_time = 0
    for i in  range(int(10* param.ts), int(10*param.te) ):
        xc = car_e.get_x()
        solver.update(xc, (traj_sl, traj_vl, traj_al))
        solver.solve() 
        traj_s,traj_v, traj_at, traj_a, traj_ab = solver.get_traj()

        sim_time += param.dt
        global_time += param.dt
        

    # cars = [car1]
    # total_time = 50.0
    # time_step = 0.1

    # simulate(road, cars, total_time, time_step)

if __name__ == '__main__':
    main()