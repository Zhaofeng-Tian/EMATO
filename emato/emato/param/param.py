import numpy as np

class Param:
    def __init__(self, cycle = 'HWFET'):
        # Solver type, choose "QP", "QP_CA", "SQP",  "NLP"
        # self.solver_type = "QP_CA"
        self.solver_type = "NLP"
        # speed profile: 'HWFET', 'EUDC'
        self.cycle = cycle
        self.car_type = "car"

        self.gd_profile_type = "flat"
        self.use_gd_prediction =True
        # Simulation global settings
        self.prediction_time = 10
        self.dt = 0.1
        self.dinit = 50
        # ACC setting
        self.T = 1.5 # Time headway in ACC process
        self.dmin = 10; self.dmax = 200.

        if self.cycle == 'HWFET':
            self.ts = 12
            self.te = 750
        elif self.cycle == 'EUDC':
            self.ts = 36
            self.te = 380
        elif self.cycle == 'INDIA_HWY':
            self.ts = 20
            self.te = 870 
        elif self.cycle == 'INDIA_URBAN':
            self.ts = 25
            self.te = 870
        elif self.cycle == 'MANHATTAN':
            self.ts = 30
            self.te = 750
        elif self.cycle == 'NYCC':
            self.ts = 70
            self.te = 570
        elif self.cycle == 'NYCTRUCK':
            self.ts = 160
            self.te = 1000
            
        self. N = int(self.prediction_time / self.dt)
        self.frmin = 0.052 # idoling fuel rate
        # QP solver weights
        self.w1 = 0.1
        self.w2 = 5
        # NLP
        self.w3 = 2.0

        self.jerkmax = 10

        # SQP 
        self.max_iteration = 10
       

        # Car dynamics
        self.vr = 27.0
        self.xr = np.array([0., 25.0])  # Reference velocity 25m/s

        self.bmin = 0.
        self.bmax = 5.
        self.vmin = 0.0
        self.vmax = 27.0

        self.xmin = np.array([-np.inf, self.vmin])  # Min state constraints
        self.xmax = np.array([np.inf, self.vmax])  # Max state constraints
        self.umin = 0.  # Min control input
        self.umax = 9.    # Max control input
        if self.solver_type == 'QP':
            self.umin = np.array([0.])
            self.umax = np.array([9.])


        # Dynamic & Fuel model
        self.k = np.array([0.0003946666666666667, 0.14715, 9.81]) # k1,k2,k3
        self.c = np.array([ 0.07224, 0.09681, 0.001075 ])         # c0,c1,c2
        self.o = np.array([0.146269884, 0.0102544085, -0.00092819697, 2.154232e-05, -4.242666666666667e-07])  # o0,o1,o2,o3,o4
        self.b = np.array([0.1569, 0.0245, -0.0007415, 0.00005975])

class TruckParam:
    def __init__(self, cycle = 'HWFET'):
        self.car_type = "truck"
        # Solver type, choose "QP", "NLP"
        self.solver_type = "NLP"
        self.gd_profile_type = "flat"
        # speed profile: 'HWFET', 'EUDC'
        self.cycle = cycle
        self.use_gd_prediction =True
        # Simulation global settings
        self.prediction_time = 5
        self.dt = 0.1
        self.dinit = 50
        if self.cycle == 'HWFET':
            self.ts = 12
            self.te = 750
        elif self.cycle == 'EUDC':
            self.ts = 36
            self.te = 380
        elif self.cycle == 'INDIA_HWY':
            self.ts = 20
            self.te = 870 
        elif self.cycle == 'INDIA_URBAN':
            self.ts = 25
            self.te = 870
        elif self.cycle == 'MANHATTAN':
            self.ts = 30
            self.te = 750
        elif self.cycle == 'NYCC':
            self.ts = 70
            self.te = 570
        elif self.cycle == 'NYCTRUCK':
            self.ts = 160
            self.te = 1000

        self. N = int(self.prediction_time / self.dt)
        self.frmin = 0.058 # idoling fuel rate
        self.jerkmax = 5
        # QP solver weights
        self.w1 = 0.1
        self.w2 = 5.0
        # NLP
        self.w3 = 10.0
        # SQP 
        self.max_iteration = 10



        # ACC setting
        self.T = 1.5 # Time headway in ACC process
        self.dmin = 50; self.dmax =300
        

        # Car dynamics
        self.vr = 25.0
        self.xr = np.array([0., 25.0])  # Reference velocity 25m/s

        self.bmin = 0.
        self.bmax = 5.
        self.vmin = 0.0
        self.vmax = 27.0

        self.xmin = np.array([-np.inf, self.vmin])  # Min state constraints
        self.xmax = np.array([np.inf, self.vmax])  # Max state constraints
        self.umin = 0.  # Min control input
        self.umax = 3.    # Max control input

        M, Av, rho, Cd, mu, g = 4800, 2.5, 1.184, 0.6, 0.006, 9.81
        # wheel_r = 0.5
        # gear_list = [7.59, 5.06, 3.38, 2.25, 1.5, 1.0, 0.75]
        # final_drive = 3.1; trans_eff = 0.92
        k1 = 0.5*Cd*rho*Av/M ; k2 = mu*g; k3 = g
        self.k = np.array([k1,k2,k3])
        self.o = np.array([3.35052784e-01, 9.09017093e-03, 3.75745106e-08, 3.49357492e-8,6.97402869e-08])
        self.c = np.array([1.65498859e-01, 3.60700787e-01, 2.42297665e-04])

        # # Dynamic & Fuel model
        # self.k = np.array([0.0003946666666666667, 0.14715, 9.81]) # k1,k2,k3
        # self.c = np.array([ 0.07224, 0.09681, 0.001075 ])         # c0,c1,c2
        # self.o = np.array([0.146269884, 0.0102544085, -0.00092819697, 2.154232e-05, -4.242666666666667e-07])  # o0,o1,o2,o3,o4
        # self.b = np.array([0.1569, 0.0245, -0.0007415, 0.00005975])

if __name__ == "__main__":
    p = Param()
    print(p.N)