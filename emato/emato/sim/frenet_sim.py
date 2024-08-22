from time import time
import numpy as np
from emato.param.param import TruckParam
from emato.obj.road.road import Road
from emato.obj.car.car import Car, Car_J
from emato.obj.car.traffic_car import TrafficCar
from emato.util.util import get_leading_profile, local_approx, get_gradient_profile, get_traj_theta
from emato.alg.nlp import NLP, NLP_J
from emato.alg.bnlp import BNLP, BNLP_R
from emato.alg.quintic import Quintic_1d, Quintic_1d_R
from emato.util.recorder import Recorder
from emato.util.plot import plot_cars
import pickle
import matplotlib.pyplot as plt


def main():
    dt = 0.1
    wx = [0.0, 350.0, 700.0, 1300.0, 1500.0]
    wy = [0.0, 500.0, 150.0, 65.0, 0.0]

    road = Road(wx, wy, num_lanes=3)
    bs = 300 # base s
    car1 = TrafficCar(lane=1, sd= [bs+30,0], speed=13.0, road = road)
    car2 = TrafficCar(lane=0, sd = [bs+0,0], speed=12.0 ,road = road)
    car3 = TrafficCar(lane=0, sd = [bs+20,0], speed=12.0 ,road = road)
    car4 = TrafficCar(lane=0, sd = [bs+45,0], speed=12.0 ,road = road)
    car5 = TrafficCar(lane=2, sd = [bs+0,0], speed=10.0 ,road = road)
    car6 = TrafficCar(lane=2, sd = [bs+20,0],speed=11.0 ,road = road)
    car7 = TrafficCar(lane=2, sd = [bs+50,0],speed=12.0 ,road = road)


    tcars = [car1, car2, car3, car4, car5, car6, car7]
    # car1 = Car(lane=0, sd= [bs+30,0], speed=13.0, road = road)
    # cars = [car1]
    total_time = 100.0

    fig, ax = plt.subplots(figsize=(10, 5))
    window_width = 30
    
    for t in np.arange(0, total_time, dt):
        ax.clear()
        road.plot_road(ax)
        for car in tcars:
            car.traffic_update_position(dt)
        print("car1 s,d: ", tcars[0].s, tcars[0].d)
        plot_cars(ax, tcars)
        ax.set_xlim([tcars[0].pose_history[-1][0]-window_width, tcars[0].pose_history[-1][0]+window_width])
        ax.set_ylim([tcars[0].pose_history[-1][1]-window_width, tcars[0].pose_history[-1][1]+window_width])
        ax.set_aspect('equal')
        ax.set_title(f"Highway Simulation at t={t:.2f} s")
        plt.pause(0.01)

    plt.show()

if __name__ == "__main__":
    main()
