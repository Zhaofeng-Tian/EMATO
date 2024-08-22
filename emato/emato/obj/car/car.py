import numpy as np
# from util.util import *
from emato.param.param import Param
class Traj:
    def __init__(self):
        self.s = 0.0
        self.v = 0.0
        self.a = 0.0
        self.jerk = 0.0
        self.at = 0.0
        self.ab = 0.0
        self.ar = 0.0
        self.fr = 0.0
        self.fc = 0.0
        self.theta = 0.0
class Car:
    def __init__(self, param):
        self.s = 0.
        self.v = 0.
        self.av = 0. ; self.pav = None # previous av
        self.at = 0.
        self.ab = 0.
        self.ar = 0.
        self.fr = 0.
        self.fc = 0.
        self.theta = 0.
        self.k = param.k
        self.c = param.c
        self.o = param.o
        self.dt = param.dt
        self.atmin = param.umin
        self.frmin = param.frmin
        self.jerkmax_t  = param.jerkmax*self.dt
        # Trajectory related properties
        # self.Traj_s = None
        # self.Traj_v = None
        # self.Traj_a = None
        # self.Traj_j = None
        # self.Traj_at = None
        # self.Traj_ar = None
        # self.Traj_fr = None
        # self.Traj_fc = None

    # Leading car
    def set_state(self, s, v, av, theta,fc):
        self.s = s
        self.v = v
        self.av = av
        self.theta = theta
        self.update_at_ab_ar()
        print(" setting leding car calc_fuel")
        self.update_fr() 
        self.fc = fc
    
    def set_theta(self, theta):
        self.theta = theta

    # Ego car
    # For QP:
    def set_states_with_av(self,av):
        """
        Only av available for updating the states
        """
        if self.pav == None:
            self.pav = av
        if av > self.pav + self.jerkmax_t:
            self.av = self.pav + self.jerkmax_t
        elif av < self.pav - self.jerkmax_t:
            self.av = self.pav - self.jerkmax_t
        else:
            self.av = av
        self.pav = self.get_av()
        self.av = av
        self.update_at_ab_ar()
        print(" setting av  calc_fuel")
        self.update_fr()

    def update_at_ab_ar(self):
        ar = self.get_ar()
        # av = at - ar -ab -> at-ab = av+ar   -> at = av+ar + ab
        if self.av >= 0:
            self.ab= 0.
            self.at = max(self.atmin, self.av + ar)
        else:
            if self.av < -ar:
                self.ab = -self.av - ar
                self.at = self.atmin
            else: 
                self.ab = 0.
                self.at = max(self.atmin, self.av + ar)
        self.ar = ar

    # For NLP:
    def set_at_av_ab(self, at, av, ab):
        if self.pav == None:
            self.pav = av
        if av > self.pav + self.jerkmax_t:
            self.av = self.pav + self.jerkmax_t
        elif av < self.pav - self.jerkmax_t:
            self.av = self.pav - self.jerkmax_t
        else:
            self.av = av
        self.pav = self.get_av()
        # self.av = av
        self.at = at
        self.ab = ab
        # self.ar = self.get_ar()
        self.update_at_ab_ar()
        print(" setting at av calc_fuel")
        self.update_fr()


    def update_fr(self):
        c0, c1, c2 = self.c
        o0,o1,o2,o3,o4 = self.o
        fr = o0 + o1*self.v + o2*self.v**2 + o3*self.v**3 + o4*self.v**4 + \
            (c0 + c1*self.v + c2*self.v**2)* self.at
        self.fr = max(self.frmin, fr)

    def get_ar(self):
        k1,k2,k3 = self.k
        ar = k1*self.v**2 + k2*np.cos(self.theta) + k3*np.sin(self.theta)
        return ar

    def step(self):
        self.s = self.s + self.v*self.dt + 0.5*self.av*self.dt**2
        self.v = self.v + self.av*self.dt
        self.fc += self.fr * self.dt

    def get_state(self):
        return self.s, self.v, self.av,  self.at, self.ar, self.fr, self.fc, self.theta

    def get_x(self):
        return np.array([self.s, self.v, self.av,  self.at, self.ar, self.fr, self.fc, self.theta])
    
    def get_s(self):
        return self.s

    def get_av(self):
        return self.av
    
# Car with jerk    
class Car_J:
    def __init__(self, param):
        self.s = 0.
        self.v = 0.
        self.av = 0. ; self.pav = None # previous av
        self.jerk = 0.
        self.at = 0.
        self.ab = 0.
        self.ar = 0.
        self.fr = 0.
        self.fc = 0.
        self.theta = 0.
        self.k = param.k
        self.c = param.c
        self.o = param.o
        self.dt = param.dt
        self.atmin = param.umin
        self.frmin = param.frmin
        # jerk max_t is a derivation of acceleration
        self.jerkmax_t  = param.jerkmax*self.dt
    # Leading car
    def set_state(self, s, v, av, jerk, theta,fc):
        self.s = s
        self.v = v
        self.av = av
        self.jerk = jerk
        self.theta = theta
        self.update_at_ab_ar()
        print(" setting leding car calc_fuel")
        self.update_fr() 
        self.fc = fc
    
    def set_theta(self, theta):
        self.theta = theta

    # Ego car
    # For QP:
    def set_states_with_av(self,av):
        """
        Only av available for updating the states
        """
        if self.pav == None:
            self.pav = av
        if av > self.pav + self.jerkmax_t:
            self.av = self.pav + self.jerkmax_t
        elif av < self.pav - self.jerkmax_t:
            self.av = self.pav - self.jerkmax_t
        else:
            self.av = av
        self.pav = self.get_av()
        self.av = av
        self.update_at_ab_ar()
        print(" setting av  calc_fuel")
        self.update_fr()

    def set_av_jerk(self,av, jerk):
        """
        Only av available for updating the states
        """
        self.jerk = jerk
        if self.pav == None:
            self.pav = av
        if av > self.pav + self.jerkmax_t:
            self.av = self.pav + self.jerkmax_t
        elif av < self.pav - self.jerkmax_t:
            self.av = self.pav - self.jerkmax_t
        else:
            self.av = av
        self.pav = self.get_av()
        self.av = av
        self.update_at_ab_ar()
        print(" setting av  calc_fuel")
        self.update_fr()


    def update_at_ab_ar(self):
        ar = self.get_ar()
        # av = at - ar -ab -> at-ab = av+ar   -> at = av+ar + ab
        if self.av >= 0:
            self.ab= 0.
            self.at = max(self.atmin, self.av + ar)
        else:
            if self.av < -ar:
                self.ab = -self.av - ar
                self.at = self.atmin
            else: 
                self.ab = 0.
                self.at = max(self.atmin, self.av + ar)
        self.ar = ar

    # For NLP:
    def set_at_av_ab(self, at, av, ab):
        if self.pav == None:
            self.pav = av
        if av > self.pav + self.jerkmax_t:
            self.av = self.pav + self.jerkmax_t
        elif av < self.pav - self.jerkmax_t:
            self.av = self.pav - self.jerkmax_t
        else:
            self.av = av
        self.pav = self.get_av()
        # self.av = av
        self.at = at
        self.ab = ab
        # self.ar = self.get_ar()
        self.update_at_ab_ar()
        print(" setting at av calc_fuel")
        self.update_fr()

    def set_at_av_ab_jerk(self, at, av, ab,jerk):
        self.jerk = jerk
        if self.pav == None:
            self.pav = av
        if av > self.pav + self.jerkmax_t:
            self.av = self.pav + self.jerkmax_t
        elif av < self.pav - self.jerkmax_t:
            self.av = self.pav - self.jerkmax_t
        else:
            self.av = av
        self.pav = self.get_av()
        # self.av = av
        self.at = at
        self.ab = ab
        # self.ar = self.get_ar()
        self.update_at_ab_ar()
        print(" setting at av calc_fuel")
        self.update_fr()


    def update_fr(self):
        c0, c1, c2 = self.c
        o0,o1,o2,o3,o4 = self.o
        fr = o0 + o1*self.v + o2*self.v**2 + o3*self.v**3 + o4*self.v**4 + \
            (c0 + c1*self.v + c2*self.v**2)* self.at
        self.fr = max(self.frmin, fr)

    def get_ar(self):
        k1,k2,k3 = self.k
        ar = k1*self.v**2 + k2*np.cos(self.theta) + k3*np.sin(self.theta)
        return ar

    def step(self): # state transition
        self.s = self.s + self.v*self.dt + 0.5*self.av*self.dt**2
        self.v = self.v + self.av*self.dt
        self.av = self.av + self.jerk * self.dt
        self.fc += self.fr * self.dt

    def get_state(self):
        return self.s, self.v, self.av, self.jerk, self.at, self.ar, self.fr, self.fc, self.theta

    def get_x(self):
        return np.array([self.s, self.v, self.av, self.jerk, self.at, self.ar, self.fr, self.fc, self.theta])
    
    def get_s(self):
        return self.s

    def get_av(self):
        return self.av

        


if __name__ == "__main__":
    param = Param()
    car = Car(param)
    s, v, av,  at, ar, fr, fc, theta = car.get_state()
    s = car.s
    car.s = 1
    print(s)
    print(car.s)